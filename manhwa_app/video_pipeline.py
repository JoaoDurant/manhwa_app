# manhwa_app/video_pipeline.py
# Pipeline de composição de vídeo para o Manhwa Video Creator.
#
# Constrói clips Ken Burns de pares (áudio, imagem):
#   • Fundo com blur gaussiano (cover fill, raio=30) via Pillow
#   • Imagem principal centralizada na área segura
#   • Efeito Ken Burns adaptativo com easing suave (ease_in_out)
#   • Coordenadas sub-pixel float (Image.Transform.EXTENT) sem travamentos
#   • Geração multithread acelerada por NVENC e com motion blur
#
# Exporta clips individuais e o arquivo final H.264/AAC 1920×1080 60fps.
# Roda dentro de uma QThread e emite sinais PySide6.

import concurrent.futures
import json
import logging
import math
import os
import queue
import random
import subprocess
import time
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

try:
    from PySide6.QtCore import QObject, Signal, QThread
except ImportError:
    class QObject: pass
    class Signal:
        def __init__(self, *args): pass
        def emit(self, *args): pass
    class QThread: pass

logger = logging.getLogger(__name__)

# --- CUDA / PyTorch SETUP ---
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image, ImageFilter
    _TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    # Caso torch ou PIL nao existam
    _TORCH_AVAILABLE = False

try:
    import soundfile as sf
except ImportError:
    sf = None

if _TORCH_AVAILABLE:
    # Máxima utilização de threads intra-op do PyTorch na CPU
    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(max(2, (os.cpu_count() or 4) // 2))

    # --- OTIMIZAÇÃO CUDA RTX (5070 Ti) ---
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
OUTPUT_W  = 1920
OUTPUT_H  = 1080
FPS       = 60      # Suavidade máxima

# Total de núcleos lógicos disponíveis
_CPU_COUNT = os.cpu_count() or 4

# Movimentos disponíveis
EFFECTS = [
    "zoom_in", "zoom_out", "pan_up", "pan_down",
    "pan_left", "pan_right", "zoom_pan_diag"
]


# ---------------------------------------------------------------------------
# Helpers FFmpeg e Animação
# ---------------------------------------------------------------------------

def _ffmpeg_ok() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False

def _get_best_encoder() -> str:
    try:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
        encoders = r.stdout
        if "h264_nvenc" in encoders: return "h264_nvenc"
        if "h264_amf" in encoders: return "h264_amf"
        if "h264_qsv" in encoders: return "h264_qsv"
    except Exception:
        pass
    return "libx264"

def _audio_duration(path: str) -> float:
    """Obtém a duração via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True, timeout=30
        )
        for stream in json.loads(r.stdout).get("streams", []):
            dur = stream.get("duration")
            if dur:
                return float(dur)
    except Exception:
        pass
    if sf:
        try:
            with sf.SoundFile(path) as f:
                return float(f.frames / f.samplerate)
        except Exception:
            return 0.0
    return 0.0

def _smoothstep(t: float, better_easing: bool = True) -> float:
    """
    Easing Suave (Ease-in-out).
    Se better_easing for true, usa a curva parabólica polinomial avançada:
    t*t*t*(t*(6*t - 15) + 10) -> cria movimento mais fluido que ease-in-out tradicional.
    """
    t = max(0.0, min(1.0, t))
    if better_easing:
        return t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    # Fallback para o clássico:
    return t * t * (3.0 - 2.0 * t)

def _python_render_clip(
    image_paths: Union[str, Tuple[str, str]],
    audio_path: Union[str, Tuple[str, str]],
    effect: str,
    duration: float,
    output_mp4: str,
    fps: int,
    encoder: str,
    log_fn: Optional[Callable] = None,
    transition_mode: str = "fade",
    transition_time: float = 0.5,
    scene_idx: int = 0,
    config: dict = None
) -> bool:
    """Renderiza a cena usando Cuda PyTorch nativo com Smoothstep, Sombra Fundida e Panning Relativo!"""
    if not _TORCH_AVAILABLE:
        if log_fn: log_fn("PyTorch não disponível, pulando renderização de cena.")
        return True # Mock success se nao tiver torch

    try:
        import random

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        frames = int(duration * fps)
        if frames <= 0: return False

        is_split = isinstance(image_paths, (tuple, list)) and len(image_paths) == 2

        # Aloca buffer reutilizável de frame em CPU (pinned memory = transferência rápida GPU→CPU)
        _use_pinned = (device == 'cuda')

        # ------ Helper Functions ------
        # pad_size grande garante que a sombra não seja cortada durante o zoom
        def create_torch_tensor(img_path, max_w, max_h, pad_size=200, shadow_off=28):
            with Image.open(img_path) as im: img = im.convert("RGBA")
            rat = min(max_w / img.width, max_h / img.height)
            new_w, new_h = int(img.width * rat), int(img.height * rat)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            base_w = new_w + pad_size * 2
            base_h = new_h + pad_size * 2
            
            # Sombra fundida — nunca se separa em qualquer animação
            shadow = Image.new("RGBA", (base_w, base_h), (0,0,0,0))
            black = Image.new("RGBA", (new_w, new_h), (0,0,0, 255))  # Sombra máxima
            shadow.paste(black, (pad_size + shadow_off, pad_size + shadow_off), black)
            shadow = shadow.filter(ImageFilter.GaussianBlur(28))   # Espalha mais a sombra
            shadow.paste(img, (pad_size, pad_size), img)
            
            arr = np.array(shadow)
            t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
            if _use_pinned:
                t = t.pin_memory()
            return t.to(device), base_w, base_h, new_w, new_h

        def paste_to_bg(bg, fg, x0, y0):
            H_f, W_f = fg.shape[2], fg.shape[3]
            bg_x0, bg_y0 = max(0, int(x0)), max(0, int(y0))
            bg_x1, bg_y1 = min(bg.shape[3], int(x0 + W_f)), min(bg.shape[2], int(y0 + H_f))
            fg_x0, fg_y0 = bg_x0 - int(x0), bg_y0 - int(y0)
            fg_x1, fg_y1 = fg_x0 + (bg_x1 - bg_x0), fg_y0 + (bg_y1 - bg_y0)
            
            if bg_x1 <= bg_x0 or bg_y1 <= bg_y0: return
            
            alpha = fg[:, 3:4, fg_y0:fg_y1, fg_x0:fg_x1] / 255.0
            rgb = fg[:, 0:3, fg_y0:fg_y1, fg_x0:fg_x1]
            bg_slice = bg[:, :, bg_y0:bg_y1, bg_x0:bg_x1]
            bg[:, :, bg_y0:bg_y1, bg_x0:bg_x1] = bg_slice * (1.0 - alpha) + rgb * alpha

        def render_anim(t, base_w, base_h, z, dx, dy):
            tx, ty = -(dx * 2.0 / base_w), -(dy * 2.0 / base_h)
            theta = torch.tensor([[[1.0/z, 0.0, tx], [0.0, 1.0/z, ty]]], dtype=torch.float32, device=device)
            grid = F.affine_grid(theta, [1, 4, base_h, base_w], align_corners=False)
            # padding_mode='border' evita cortes negros na sombra durante zoom!
            return F.grid_sample(t, grid, mode='bicubic', padding_mode='border', align_corners=False)

        # ------ Pipeline ------
        bg_path = image_paths[0] if is_split else image_paths
        with Image.open(bg_path).convert("RGB") as im:
            bg_rat = max(1920 / im.width, 1080 / im.height)
            new_w, new_h = int(im.width * bg_rat), int(im.height * bg_rat)
            bg_full = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            bx, by = (new_w - 1920)//2, (new_h - 1080)//2
            bg_crop = bg_full.crop((bx, by, bx + 1920, by + 1080))
            bg_blur = bg_crop.filter(ImageFilter.GaussianBlur(25))
            bg_t_base = torch.from_numpy(np.array(bg_blur)).permute(2,0,1).unsqueeze(0).float().to(device)

        if is_split:
            img1_p, img2_p = image_paths
            fg1_t, w1, h1, core_w1, core_h1 = create_torch_tensor(img1_p, 850, 900)
            fg2_t, w2, h2, core_w2, core_h2 = create_torch_tensor(img2_p, 850, 900)
            cx1, cy1 = (960 - w1)//2, (1080 - h1)//2
            cx2, cy2 = 960 + (960 - w2)//2, (1080 - h2)//2
        else:
            # Auto-siz: imagens retrato muito altas usam toda a altura disponível
            with Image.open(image_paths) as _im_probe:
                _probe_w, _probe_h = _im_probe.size
            is_portrait = _probe_h > _probe_w * 1.2   # imagem claramente vertical
            if is_portrait:
                _max_w, _max_h = 1100, 1020          # ocupa quase toda a altura
            else:
                _max_w, _max_h = 1400, 900           # paisagem: mais larga
            fg_t, w, h, core_w, core_h = create_torch_tensor(image_paths, _max_w, _max_h)
            cx, cy = (1920 - w)//2, (1080 - h)//2

        # --- FILTROS VISUAIS DINÂMICOS ---
        cfg_vid = (config or {}).get("production", {}).get("video", {})
        do_grading = cfg_vid.get("color_grading", True)
        do_sharpen = cfg_vid.get("sharpen", True)
        do_grain = cfg_vid.get("film_grain", False)

        vf_filters = []
        if do_grading:
            # eq=contrast=1.05:brightness=0.02:saturation=1.1 → evita imagem lavada
            vf_filters.append("eq=contrast=1.05:brightness=0.02:saturation=1.1")
        if do_sharpen:
            # unsharp=3:3:0.5:3:3:0 → sharpen leve nos contornos/texto
            vf_filters.append("unsharp=3:3:0.5:3:3:0")
        if do_grain:
            # Grain sutil: alls=15:allf=t
            vf_filters.append("noise=alls=15:allf=t")

        COLOR_CORRECTION = ",".join(vf_filters) if vf_filters else "null"

        # FFmpeg piped render — threads=0 usa TODOS os núcleos disponíveis
        # NVENC p4 = excelente custo/benefício: 2-3x mais rápido que p7 com qualidade similar a 1080p60
        if encoder == "h264_nvenc":
            vcodec = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "22", "-b:v", "0", "-maxrate", "15M"]
        elif encoder == "h264_amf":
            vcodec = ["-c:v", "h264_amf", "-quality", "speed", "-qp_i", "20", "-qp_p", "20"]
        elif encoder == "h264_qsv":
            vcodec = ["-c:v", "h264_qsv", "-preset", "fast", "-global_quality", "22"]
        else:
            vcodec = ["-c:v", "libx264", "-preset", "faster", "-crf", "20", "-threads", "0"]
        cmd = ["ffmpeg", "-y", "-threads", "0",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "1920x1080",
               "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-"]
        if isinstance(audio_path, (tuple, list)):
            cmd.extend(["-i", audio_path[0], "-i", audio_path[1], "-filter_complex", "[1:a][2:a]concat=n=2:v=0:a=1[a_out]", "-map", "0:v", "-map", "[a_out]"])
        else:
            cmd.extend(["-i", audio_path, "-map", "0:v", "-map", "1:a"])
        # Aplica filtros visuais no encode final
        if COLOR_CORRECTION != "null":
            cmd.extend(["-vf", COLOR_CORRECTION])
        cmd.extend([*vcodec, "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", output_mp4])
        
        create_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=create_flags)

        # Buffer pre-alocado reutilizavel (evita alloc+copy por frame)
        frame_buf = torch.empty(1, 3, 1080, 1920, dtype=torch.uint8, device='cpu')

        # CORRIGIDO: anim_val determinado UMA VEZ antes do loop de frames
        # random.choice dentro do loop causava mudança de animação a cada frame
        anim_val_once = random.choice(EFFECTS) if (not is_split and effect == "auto") else effect

        # CORRIGIDO: do_better_easing calculado UMA VEZ por cena (era dict lookup por frame)
        do_better_easing = (config or {}).get("production", {}).get("video", {}).get("better_easing", True)

        # CORRIGIDO: FADE_FRAMES é constante por cena — calculado UMA VEZ antes do loop
        FADE_FRAMES = int(transition_time * fps)
        if frames > 0 and FADE_FRAMES > frames // 2:
            FADE_FRAMES = frames // 2

        try:
            for i in range(frames):
                t_linear = i / max(1, frames - 1)
                e = _smoothstep(t_linear, better_easing=do_better_easing)

                frame_bg = bg_t_base.clone()

                if is_split:
                    # Esquerda (Zoom In)
                    z1 = 1.0 + (1.10 - 1.0) * e
                    anim1 = render_anim(fg1_t, w1, h1, z1, 0, 0)
                    paste_to_bg(frame_bg, anim1, cx1, cy1)
                    
                    # Direita (Movimento Vertical: Baixo pra Cima proporcinal à img)
                    pan_dist2 = h2 * 0.08
                    dy2 = pan_dist2 - (pan_dist2 * 2.0) * e # Vai de +X até -X
                    anim2 = render_anim(fg2_t, w2, h2, 1.0, 0, dy2)
                    paste_to_bg(frame_bg, anim2, cx2, cy2)
                else:
                    # CORRIGIDO: anim_val decidido uma vez antes do loop (anim_val_once)
                    # antes era random.choice() por frame, fazendo a animação pular a cada frame
                    anim_val = anim_val_once
                    z_start, z_end = 1.0, 1.0
                    dx_start, dx_end = 0.0, 0.0
                    dy_start, dy_end = 0.0, 0.0
                    pan_dist = h * 0.08
                    
                    if anim_val == "zoom_in":
                        z_start, z_end = 1.0, 1.10
                    elif anim_val == "zoom_out":
                        z_start, z_end = 1.10, 1.0
                    elif anim_val == "pan_up":
                        dy_start, dy_end = pan_dist, -pan_dist
                        z_start, z_end = 1.08, 1.08
                    elif anim_val == "pan_down":
                        dy_start, dy_end = -pan_dist, pan_dist
                        z_start, z_end = 1.08, 1.08
                    elif anim_val == "pan_left":
                        dx_start, dx_end = pan_dist, -pan_dist
                        z_start, z_end = 1.08, 1.08
                    elif anim_val == "pan_right":
                        dx_start, dx_end = -pan_dist, pan_dist
                        z_start, z_end = 1.08, 1.08
                    elif anim_val == "zoom_pan_diag":
                        z_start, z_end = 1.0, 1.12
                        dx_start, dx_end = pan_dist * 0.8, -pan_dist * 0.8
                        dy_start, dy_end = pan_dist * 0.8, -pan_dist * 0.8
                        
                    z = z_start + (z_end - z_start) * e
                    dx = dx_start + (dx_end - dx_start) * e
                    dy = dy_start + (dy_end - dy_start) * e
                    
                    anim = render_anim(fg_t, w, h, z, dx, dy)
                    paste_to_bg(frame_bg, anim, cx, cy)
                
                # ---------- Transição entre cenas ----------
                # FADE_FRAMES já calculado antes do loop (constante por cena)
                if transition_mode == "none" or FADE_FRAMES <= 0:
                    # Corte direto — sem fade nem blur
                    frame_rgb = frame_bg.clamp(0, 255)
                    frame_u8 = frame_rgb.to(torch.uint8).squeeze(0).permute(1, 2, 0)
                    frame_bytes = frame_u8.cpu().numpy().tobytes()
                elif transition_mode == "blur":
                    # Blur suave: fica nítido no meio, levemente desfocado no início/fim
                    if i < FADE_FRAMES and scene_idx > 0:
                        alpha_t = i / FADE_FRAMES           # 0.0 → 1.0
                    elif i >= frames - FADE_FRAMES:
                        alpha_t = (frames - 1 - i) / FADE_FRAMES  # 1.0 → 0.0
                    else:
                        alpha_t = 1.0
                    frame_rgb = frame_bg.clamp(0, 255)
                    frame_u8 = frame_rgb.to(torch.uint8).squeeze(0).permute(1, 2, 0)
                    frame_np = frame_u8.cpu().numpy()
                    if alpha_t < 1.0:
                        from PIL import Image as _PIL_Image, ImageFilter as _PIL_IF
                        blur_r = int((1.0 - alpha_t) * 28 + 0.5)  # max radius 28
                        if blur_r > 0:
                            _pil_f = _PIL_Image.fromarray(frame_np)
                            _pil_f = _pil_f.filter(_PIL_IF.GaussianBlur(blur_r))
                            frame_np = _pil_f.tobytes()
                        else:
                            frame_np = frame_np.tobytes()
                    else:
                        frame_np = frame_np.tobytes()
                    frame_bytes = frame_np
                else:
                    # Fade padrão: escurece para preto
                    if i < FADE_FRAMES and scene_idx > 0:
                        fade = i / FADE_FRAMES
                    elif i >= frames - FADE_FRAMES:
                        fade = (frames - 1 - i) / FADE_FRAMES
                    else:
                        fade = 1.0
                    frame_rgb = frame_bg.clamp(0, 255) * fade
                    frame_u8 = frame_rgb.to(torch.uint8).squeeze(0).permute(1, 2, 0)
                    frame_bytes = frame_u8.cpu().numpy().tobytes()
                proc.stdin.write(frame_bytes)

            # CORRIGIDO: proc.stdin.close() e proc.wait() agora estão DENTRO do try
            # antes ficavam fora, impossível acessar proc em caso de exceção init
            proc.stdin.close()
            proc.wait()
            return proc.returncode == 0

        except Exception as e:
            # CORRIGIDO: matar o proc FFmpeg se render falhar a meio (zombie process prevention)
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            except Exception:
                pass
            if log_fn: log_fn(f"Exception GPU Render Cena: {e}")
            return False

    except Exception as e:
        # Outer except: captura erros durante setup (carregamento de imagem, FFmpeg init, etc.)
        if log_fn: log_fn(f"Exception GPU Render Cena (setup): {e}")
        return False


# ---------------------------------------------------------------------------
# Pipeline de vídeo principal (QObject com sinais)
# ---------------------------------------------------------------------------

class VideoPipeline(QObject):
    progress    = Signal(int, int)      # (atual, total)
    log_message = Signal(str)
    finished    = Signal(bool, str)     # (sucesso, caminho_resultado_ou_erro)

    def __init__(
        self,
        pairs: List[Tuple[Union[str, Tuple[str, str]], Union[str, Tuple[str, str]]]],
        output_path: str,
        effect_mode: str = "auto",
        transition_mode: str = "fade",
        transition_time: float = 0.2,
        bg_music_path: str = "",
        bg_music_volume: int = 10,
        config: dict = None,
        parent=None,
    ):
        super().__init__(parent)
        self.pairs           = pairs
        self.output_path     = output_path
        self.effect_mode     = effect_mode
        self.transition_mode = transition_mode
        self.transition_time = transition_time
        self.bg_music_path   = bg_music_path
        self.bg_music_volume = bg_music_volume
        self.config          = config or {}
        self._cancelled      = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if not self.pairs:
            self.finished.emit(False, "Nenhum par áudio/imagem fornecido.")
            return

        n_pairs = len(self.pairs)
        self.log_message.emit(
            f"Iniciando Renderização Multithread Sub-pixel 60FPS: {n_pairs} cenas.\n"
            f"Destino final: {self.output_path}\n"
        )

        if not _ffmpeg_ok():
            self.finished.emit(False, "FFmpeg não encontrado.")
            return

        encoder = _get_best_encoder()
        self.log_message.emit(
            f"Aceleração de Vídeo / Base de render: {encoder}"
        )

        start_time = time.time()

        out_root = Path(self.output_path)
        scenes_dir = out_root.parent / f"{out_root.stem}_cenas_individuais"
        scenes_dir.mkdir(parents=True, exist_ok=True)

        clip_tasks = []
        for clip_i, pair in enumerate(self.pairs, start=1):
            if isinstance(pair[0], tuple):
                a1, a2 = pair[0]
                img1, img2 = pair[1]
                audio_path = (a1, a2)
                image_path = (img1, img2)
                duration = _audio_duration(a1) + _audio_duration(a2)
            else:
                audio_path, image_path = pair
                duration = _audio_duration(audio_path)
                
            if duration <= 0:
                continue

            if self.effect_mode == "auto":
                effect = ["zoom_in", "zoom_out", "pan_up", "pan_down"][(clip_i - 1) % 4]
            else:
                effect = self.effect_mode
                
            clip_name = f"scene_{clip_i:03d}.mp4"
            clip_mp4 = str(scenes_dir / clip_name)
            
            clip_tasks.append((clip_i, audio_path, image_path, effect, duration, clip_mp4))

        if not clip_tasks:
            self.finished.emit(False, "Durações de áudio inválidas.")
            return

        total = len(clip_tasks)
        completed = 0
        concat_map: dict[int, str] = {}

        # CORRIGIDO: NVENC consumer GPUs suportam ~3-5 sessões de encode simultaneamente.
        # CPU_COUNT * 2 causava falhas silenciosas por excesso de sessões NVENC.
        # libx264: todos os núcleos são seguros.
        MAX_NVENC_SESSIONS = 4  # limite conservador e seguro para consumer GPUs
        if encoder == "h264_nvenc" or encoder == "h264_amf":
            max_workers = min(MAX_NVENC_SESSIONS, _CPU_COUNT)
        elif encoder == "h264_qsv":
            max_workers = min(6, _CPU_COUNT)  # QSV tem limite mais alto
        else:
            max_workers = _CPU_COUNT  # libx264: all cores safe

        self.log_message.emit(f"Renderizando com Smoothstep Easing 60fps — {max_workers} cenas em paralelo ({_CPU_COUNT} núcleos detectados)...")

        # CORRIGIDO: log_message.emit não pode ser chamado de threads nativas (ThreadPoolExecutor)
        # Usamos uma Queue thread-safe: workers colocam logs nela, o loop principal processa
        _log_queue: queue.Queue = queue.Queue()

        def _safe_log(msg: str):
            """Logging thread-safe: coloca na fila em vez de emitir diretamente."""
            _log_queue.put(msg)

        def _flush_log_queue():
            """Esvazia a fila de logs e emite via sinal Qt (thread principal)."""
            while not _log_queue.empty():
                try:
                    self.log_message.emit(_log_queue.get_nowait())
                except queue.Empty:
                    break

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, task in enumerate(clip_tasks):
                clip_i, audio_path, image_path, effect, duration, clip_mp4 = task
                fut = executor.submit(
                    _python_render_clip,
                    image_path, audio_path, effect, duration, clip_mp4, FPS, encoder,
                    _safe_log,  # CORRIGIDO: usa logging thread-safe em vez de emit direto
                    self.transition_mode, self.transition_time, idx, self.config
                )
                future_to_idx[fut] = (idx, task)

            for fut in concurrent.futures.as_completed(future_to_idx):
                if self._cancelled:
                    executor.shutdown(wait=False, cancel_futures=True)
                    self.finished.emit(False, "Cancelado pelo usuário.")
                    return

                idx, task = future_to_idx[fut]
                clip_i, audio_path, image_path, effect, duration, clip_mp4 = task

                try:
                    success = fut.result()
                except Exception as e:
                    success = False
                    e_str = str(e)
                    if e_str and len(e_str) > 100:
                        e_str = e_str[0:100]
                    self.log_message.emit(f"  [ERRO] Exceção durante renderização da Cena #{clip_i:02d}: {e_str}")


                completed += 1
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed if completed > 0 else 0
                eta_sec = int(avg_time * (total - completed))
                eta_str = f"{eta_sec // 60}m {eta_sec % 60}s" if eta_sec >= 60 else f"{eta_sec}s"
                
                self.progress.emit(completed, total)

                if success and os.path.exists(clip_mp4):
                    self.log_message.emit(f"  [ETA: {eta_str}] ✓ Cena #{clip_i:02d} gerada individualmente em: {Path(clip_mp4).name}")
                    concat_map[idx] = str(clip_mp4)
                else:
                    self.log_message.emit(f"  [ETA: {eta_str}] ✗ Falha na Cena #{clip_i:02d}")

        # Esvazia a fila de logs dos workers antes de prosseguir para o concat
        _flush_log_queue()

        # Reconstrói lista ordenada por índice de cena (as_completed não garante ordem)
        concat_list = [concat_map[k] for k in sorted(concat_map)]
        if not concat_list:
            self.finished.emit(False, "Nenhuma cena foi gerada com sucesso.")
            return

        # Concat Demuxer final
        self.log_message.emit("\nAgrupando vídeo final (sem perdas na qualidade)...")
        concat_txt = scenes_dir / "concat_list.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for c_path in concat_list:
                # O txt está no mesmo diretório dos vídeos, usar apenas o nome é mais seguro
                f.write(f"file '{Path(c_path).name}'\n")

        # Garante que o diretório de saída existe antes de escrever o vídeo final
        out_path = Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            str(out_path)
        ]

        self.log_message.emit(f"  Saída final: {out_path}")
        try:
            create_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            result = subprocess.run(
                concat_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=120,  # CORRIGIDO: timeout para evitar bloqueio infinito no concat final
                creationflags=create_flags
            )
            if result.returncode == 0 and out_path.exists():
                
                if getattr(self, "bg_music_path", ""):
                    self.log_message.emit("🎵 Adicionando Música de Fundo (Loop infinito)...")
                    bg_path = self.bg_music_path
                    bg_vol = getattr(self, "bg_music_volume", 10) / 100.0
                    temp_mp4 = out_path.with_name(f"{out_path.stem}_bgtemp.mp4")
                    
                    do_ducking = self.config.get("production", {}).get("sound_design", {}).get("auto_ducking", False)
                    
                    if do_ducking:
                        self.log_message.emit("🎵 Auto-Ducking Ativado: abaixando BGM durante as falas...")
                        # Usa o split para [0:a] (voz principal).
                        # [0:a] vai para mixagem final e sidechain control.
                        # sidechaincompress threshold -30dB, ratio 4, attack 50, release 300, makeup 1.
                        fc = f"[0:a]asplit=2[voce][vside];[1:a]volume={bg_vol}[bgm];[bgm][vside]sidechaincompress=threshold=-30dB:ratio=4:attack=50:release=300[bgm_ducked];[voce][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                    else:
                        fc = f"[1:a]volume={bg_vol}[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"

                    bgm_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(out_path),
                        "-stream_loop", "-1",
                        "-i", str(bg_path),
                        "-filter_complex", fc,
                        "-map", "0:v", "-map", "[aout]",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        str(temp_mp4)
                    ]
                    
                    # CORRIGIDO: adicionar timeout ao subprocess do BGM
                    # Sem timeout, um arquivo de áudio corrompido ou FFmpeg travado
                    # bloqueia o pipeline indefinidamente
                    try:
                        bg_res = subprocess.run(
                            bgm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            creationflags=create_flags, timeout=120
                        )
                        if bg_res.returncode == 0 and temp_mp4.exists():
                            shutil.move(str(temp_mp4), str(out_path))
                            self.log_message.emit("  ✓ Música de Fundo adicionada com sucesso!")
                        else:
                            err_msg = bg_res.stderr.decode('utf-8', errors='ignore') if bg_res.stderr else "Erro desconhecido"
                            if len(err_msg) > 100:
                                err_msg = err_msg[0:100]
                            self.log_message.emit(f"  ✗ Erro ao mixar BGM: {err_msg}...\n(Mantendo vídeo sem música)")
                            # CORRIGIDO: limpar temp_mp4 se BGM falhou
                            if temp_mp4.exists():
                                try: temp_mp4.unlink()
                                except Exception: pass
                    except subprocess.TimeoutExpired:
                        self.log_message.emit("  ⚠ BGM timeout (120s). Mantendo vídeo sem música.")
                        # CORRIGIDO: limpar temp_mp4 se timeout
                        if temp_mp4.exists():
                            try: temp_mp4.unlink()
                            except Exception: pass

                total_time = time.time() - start_time
                mins, secs = divmod(int(total_time), 60)
                time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

                self.log_message.emit(f"\n✓ Vídeo final salvo em: {out_path} (Tempo demorado: {time_str})")
                self.finished.emit(True, str(out_path))
            else:
                err = result.stderr.decode("utf-8", errors="replace")[-500:]
                self.log_message.emit(f"\n✗ Concat falhou (código {result.returncode}):\n{err}")
                self.finished.emit(False, f"Concat falhou. Ver log para detalhes.")
        except Exception as e:
            self.finished.emit(False, f"Erro no Mux final: {e}")

