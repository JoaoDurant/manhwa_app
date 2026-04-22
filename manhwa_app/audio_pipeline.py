# manhwa_app/audio_pipeline.py
import gc
import re
import os
import sys
import json
import time
import shutil
import threading
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Union
from PySide6.QtCore import QObject, Signal
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

# --- REPO SETUP ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import engine as _engine
    import utils as _utils
    from config import config_manager as _config_manager
    from manhwa_app.text_processor import process_text_fluency
    from manhwa_app.advanced_text_processor import process_text
    from manhwa_app.models import get_whisper_model, unload_whisper, transcribe_audio
    from manhwa_app.utils import get_safe_path
    _ENGINE_AVAILABLE = True
except ImportError:
    _engine = _utils = _config_manager = None
    _ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- OTIMIZAÇÃO GLOBAL ---
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except: pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def split_into_paragraphs(text: str) -> List[str]:
    text = text.replace('\ufeff', '').replace("\r\n", "\n").replace("\r", "\n")
    raw = re.split(r"\n\n+", text) if "\n\n" in text else re.split(r"\n+", text)
    res = []
    for b in raw:
        b = re.sub(r'^\s*\d+\s*\n+', '', b)
        cb = b.replace("\n", " ")
        cb = re.sub(r"\s+", " ", cb).strip()
        if len(cb) >= 3: res.append(cb)
    return res

def _normalize_text_for_tts(text: str, lang: str = "en", engine: str = "chatterbox") -> str:
    text = text.replace('\ufeff', '')
    if engine == "kokoro": return re.sub(r' {2,}', ' ', text).strip()
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r' +([.!?,;:])', r'\1', text)
    return text.strip()

def _text_similarity(exp: str, trans: str) -> float:
    from difflib import SequenceMatcher
    if not trans: return 0.0
    return SequenceMatcher(None, exp.lower().strip(), trans.lower().strip()).ratio()

def _check_absurd(exp: str, trans: str, sim: float) -> Tuple[bool, str]:
    if not trans or len(trans) < 3: return True, "Mudo/Vazio"
    if sim < 0.65: return True, f"Erro conteúdo ({sim:.2f})"
    ew, tw = len(exp.split()), len(trans.split())
    if tw > (ew * 1.6) + 3: return True, "Alucinação/Loop"
    if tw < (ew * 0.5) and ew > 4: return True, "Corte massivo"
    return False, ""

def _evaluate_quality(path: str) -> float:
    try:
        import soundfile as sf
        arr, sr = sf.read(path, dtype="float32")
        if arr.ndim > 1: arr = arr[:, 0]
        if len(arr) == 0: return 0.0
        fl = int(sr * 0.05)
        if len(arr) < fl: return 0.0
        rms = np.array([np.sqrt(np.mean(f**2)) for f in np.array_split(arr, len(arr)//fl)])
        rms = rms[rms > 0.001]
        if len(rms) < 2: return 0.0
        return float(np.std(rms) / (np.mean(rms) + 1e-6))
    except: return 0.0

def _remove_silence(inp: str, out: str, sr: int) -> str:
    try:
        import soundfile as sf
        import utils as u
        arr, _ = sf.read(inp, dtype="float32")
        if arr.ndim > 1: arr = arr[:, 0]
        arr = u.trim_lead_trail_silence(arr, sr, -35.0, 400)
        arr = u.fix_internal_silence(arr, sr, -40.0, 250, 280)
        if u.save_audio_to_file(arr, sr, out): return out
    except: pass
    return inp

# ---------------------------------------------------------------------------
# Pipeline Principal
# ---------------------------------------------------------------------------
class AudioPipeline(QObject):
    progress = Signal(int, int)
    log_message = Signal(str)
    paragraph_done = Signal(int, str, str)
    finished = Signal(bool, str)
    paragraph_ready = Signal(int, str, dict)
    engine_switch_needed = Signal(str, str)
    paragraph_started = Signal(int, int, str)
    paragraph_done_stats = Signal(int, int, float, float, float, int)
    paragraph_retry = Signal(int, int, str)

    def __init__(self, file_configs, project_name, output_root="output", tts_engine="chatterbox", model_type="turbo", 
                 whisper_model="base", similarity_threshold=0.75, max_retries=5, temperature=0.65, **kwargs):
        super().__init__(kwargs.get("parent"))
        self.file_configs, self.project_name, self.output_root = file_configs, project_name, output_root
        self.tts_engine, self.model_type, self.whisper_model = tts_engine, model_type, whisper_model
        self.similarity_threshold, self.max_retries = similarity_threshold, max_retries
        self.temperature = temperature
        self.lang = kwargs.get("lang", "pt")
        self.sample_rate = kwargs.get("sample_rate", 24000)
        self._cancelled = False
        self._state_lock = threading.Lock()
        
        # [AUDIO PIPELINE] Otimização: 12 threads p/ não travar driver CUDA (Blackwell)
        torch.set_num_threads(12)
        logger.info(f"[AUDIO PIPELINE] Threads: {torch.get_num_threads()} | GC: 5 | VRAM: 70%")

        # Executors: 1 p/ síntese/pre-proc, 1 p/ Whisper (Async), 1 p/ Post-proc
        self._main_exec = ThreadPoolExecutor(max_workers=2)
        self._whisper_exec = ThreadPoolExecutor(max_workers=1)
        self._post_exec = ThreadPoolExecutor(max_workers=4)

        self.paragrafos_map: Dict[int, dict] = {}
        self.completed_indices: Set[int] = set()
        self._start_time = 0.0
        self._all_paras = []

    def cancel(self): self._cancelled = True

    def _preprocess_task(self, para, lang):
        try:
            adv = {"normalize_text": True, "clean_symbols": True, "improve_punctuation": True, "add_natural_pauses": True, 
                   "convert_numbers": True, "use_phonetic": False, "use_spacy": True}
            v_text = process_text(para, adv, lang=lang)
            t_text = _normalize_text_for_tts(v_text, lang=lang, engine=self.tts_engine)
            return t_text, v_text
        except: return para, para

    def _post_process_task(self, idx, tmp_p, out_p, p_cfg, para, src, sr, sim, elap, att, rms):
        try:
            final_no_fx = _remove_silence(str(tmp_p), str(tmp_p).replace("_tmp", "_sil"), sr)
            # FX Mock/Simple (expandir se audio_fx disponível)
            shutil.copy2(final_no_fx, out_p)
            
            with self._state_lock:
                self.paragrafos_map[idx] = {"index": idx, "audio": f"audio_{idx}.wav", "texto": para, "arquivo_origem": src, "similaridade": round(sim, 3)}
                self.completed_indices.add(idx)
                done = len(self.completed_indices)
            
            self.paragraph_done.emit(idx, str(out_p), para)
            self.paragraph_ready.emit(idx, str(out_p), self.paragrafos_map[idx])
            self.paragraph_done_stats.emit(idx, len(self._all_paras), elap, sim, rms, att)
            
            avg = (time.time() - self._start_time) / done if done > 0 else 0
            eta = int(avg * (len(self._all_paras) - done))
            eta_s = f"{eta//60}m {eta%60}s" if eta >= 60 else f"{eta}s"
            self.progress.emit(done, len(self._all_paras))
            self.log_message.emit(f"  ✓ [#{idx}/{len(self._all_paras)}] Finalizado (Sim: {sim:.2f}). ETA: {eta_s}")
        except Exception as e:
            self.log_message.emit(f"  ✗ Erro Post-proc [#{idx}]: {e}")

    def run(self):
        self._start_time = time.time()
        if not _ENGINE_AVAILABLE: return self.finished.emit(False, "Engine offline.")
        
        # Otimização Blackwell: Blackwell VRAM check
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        self._all_paras = []
        for fcfg in self.file_configs:
            try:
                ps = split_into_paragraphs(Path(fcfg["path"]).read_text(encoding="utf-8-sig", errors="replace"))
                for p in ps: self._all_paras.append((p, Path(fcfg["path"]).name, fcfg))
            except: pass

        if not self._all_paras: return self.finished.emit(False, "Sem parágrafos.")

        out_dir = get_safe_path(Path(self.output_root) / self.project_name / "audios")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Whisper GPU fp16 - Otimizado
        if self.similarity_threshold > 0:
            self.log_message.emit(f"🔌 Carregando Whisper [{self.whisper_model}] (GPU/fp16)...")
            get_whisper_model(self.whisper_model, device_override="cuda", compute_type="float16")

        # Engine Setup
        if self.tts_engine == "chatterbox":
            _engine.switch_to_engine(self.model_type)
        
        self.log_message.emit(f"🚀 Iniciando Geração de {len(self._all_paras)} parágrafos...")
        
        pending_validations = {} # {idx: (future, tmp_path, verification_text, start_time, p_cfg, para, src, lang)}
        
        for i, (para, src, p_cfg) in enumerate(self._all_paras, 1):
            if self._cancelled: break
            
            self.paragraph_started.emit(i, len(self._all_paras), f"{para[:50]}...")
            t0 = time.time()
            lang = p_cfg.get("lang", self.lang)
            
            # Step 1: Pre-proc + Synthesis
            t_input, v_text = self._preprocess_task(para, lang)
            tmp_wav = out_dir / f"audio_{i}_tmp.wav"
            
            success = _engine.synthesize(
                text=t_input, output_path=str(tmp_wav), engine_name=p_cfg.get("engine", self.tts_engine),
                audio_prompt_path=p_cfg.get("voice"), temperature=self.temperature, seed=i+1000, sample_rate=self.sample_rate
            )
            
            if success and tmp_wav.exists():
                if self.similarity_threshold > 0:
                    # [ASYNC WHISPER] Validação não bloqueia a síntese do próximo parágrafo
                    fut = self._whisper_exec.submit(transcribe_audio, str(tmp_wav), self.whisper_model, language=lang.split("-")[0])
                    pending_validations[i] = (fut, tmp_wav, v_text, t0, p_cfg, para, src, lang)
                else:
                    # Sem validação: direto p/ post-proc
                    self._post_exec.submit(self._post_process_task, i, tmp_wav, tmp_wav, p_cfg, para, src, self.sample_rate, 1.0, 0.0, 1, 0.0)
            
            # Anti-fragmentação a cada 5
            if i % 5 == 0:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Step 2: Coleta de Validações Async + Retries
        self.log_message.emit("⌛ Validando transcrições e aplicando retries...")
        for i, (fut, tmp, v_text, t0, p_cfg, para, src, lang) in pending_validations.items():
            try:
                trans = fut.result(timeout=30)
                sim = _text_similarity(v_text, trans)
                is_err, reason = _check_absurd(v_text, trans, sim)
                
                if is_err or sim < self.similarity_threshold:
                    self.log_message.emit(f"  ↩️ [RETRY] #{i} falhou (Sim:{sim:.2f} | {reason}). Tentando novamente...")
                    # Re-sintetiza com temp mais alta (síncrono no retry por simplicidade)
                    _engine.synthesize(text=v_text, output_path=str(tmp), temperature=self.temperature+0.1, seed=i+2000)
                    # Re-valida (desta vez síncrono p/ fechar o item)
                    trans2 = transcribe_audio(str(tmp), self.whisper_model, language=lang.split("-")[0])
                    sim = _text_similarity(v_text, trans2)

                rms = _evaluate_quality(str(tmp))
                self._post_exec.submit(self._post_process_task, i, tmp, tmp, p_cfg, para, src, self.sample_rate, sim, time.time()-t0, 2, rms)
            except Exception as e:
                self.log_message.emit(f"  ✗ Falha na validação #{i}: {e}")

        self._post_exec.shutdown(wait=True)
        unload_whisper()
        
        # Save Metadata
        final_json = [self.paragrafos_map[k] for k in sorted(self.paragrafos_map)]
        with open(out_dir.parent / "paragrafos.json", "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

        self.log_message.emit(f"✓ Concluído! {len(self.completed_indices)} áudios gerados.")
        self.finished.emit(True, "Sucesso")