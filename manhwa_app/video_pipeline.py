import concurrent.futures
import json
import logging
import os
import queue
import random
import subprocess
import time
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from manhwa_app.utils import get_safe_path

try:
    from PySide6.QtCore import QObject, Signal
except ImportError:
    class QObject: pass
    class Signal:
        def __init__(self, *args): pass
        def emit(self, *args): pass

logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

# --- CUDA / PyTorch SETUP ---
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    _TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# CORE RENDER - Top-level, sem closures, sem torch.compile (instável no Win)
# ---------------------------------------------------------------------------
def _render_anim_impl(t, base_w, base_h, z, dx, dy):
    tx = -(dx * 2.0 / base_w)
    ty = -(dy * 2.0 / base_h)
    # Grid sample espera [1, 4, H, W] para RGBA
    theta = torch.tensor([[[1.0/z, 0.0, tx], [0.0, 1.0/z, ty]]],
                          dtype=torch.float32, device=t.device)
    grid = F.affine_grid(theta, [1, 4, base_h, base_w], align_corners=False)
    return F.grid_sample(t, grid, mode='bicubic', padding_mode='border', align_corners=False)

_IMAGE_TENSOR_CACHE: dict = {}
_IMAGE_CACHE_MAX = 32

def clear_pipeline_cache():
    """Limpa o cache global de tensores de imagem para liberar VRAM."""
    global _IMAGE_TENSOR_CACHE
    _IMAGE_TENSOR_CACHE.clear()
    if _TORCH_AVAILABLE:
        import torch
        torch.cuda.empty_cache()
    logger.info("🗑️ Cache de tensores da VideoPipeline limpo.")

_warmup_done = False
_warmup_lock = threading.Lock()

def _ensure_warmup():
    """Pré-aquecimento thread-safe dentro do worker."""
    global _warmup_done
    if _warmup_done or not _TORCH_AVAILABLE: return
    with _warmup_lock:
        if _warmup_done: return
        if torch.cuda.is_available():
            try:
                dummy = torch.zeros(1, 4, 128, 128, device='cuda')
                _render_anim_impl(dummy, 128, 128, 1.0, 0.0, 0.0)
                torch.cuda.synchronize()
            except Exception: pass
        _warmup_done = True

# ---------------------------------------------------------------------------
# Constantes e Helpers
# ---------------------------------------------------------------------------
OUTPUT_W  = 1920
OUTPUT_H  = 1080
FPS       = 60
BATCH_SIZE = 60  # 1s de vídeo por escrita no stdin (reduz syscalls do pipe)

EFFECTS = ["zoom_in", "zoom_out", "pan_up", "pan_down", "pan_left", "pan_right"]

def _ffmpeg_ok() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except: return False

def _get_best_encoder() -> str:
    try:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
        if "h264_nvenc" in r.stdout: return "h264_nvenc"
        if "h264_amf" in r.stdout: return "h264_amf"
        if "h264_qsv" in r.stdout: return "h264_qsv"
    except: pass
    return "libx264"

def _audio_duration(path: str) -> float:
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if s.get("duration"): return float(s["duration"])
    except: pass
    return 0.0

def _validate_existing_clip(clip_path: str, audio_path, image_path=None, tolerance: float = 0.5) -> bool:
    if not os.path.exists(clip_path):
        return False
    clip_dur = _audio_duration(clip_path)
    if clip_dur <= 0:
        return False
    if isinstance(audio_path, (tuple, list)):
        expected = sum(_audio_duration(a) for a in audio_path)
    else:
        expected = _audio_duration(audio_path)
    if abs(clip_dur - expected) > tolerance:
        return False

    # Verifica se o clipe foi gerado com a mesma imagem via sidecar .json
    if image_path:
        sidecar = Path(clip_path).with_suffix(".json")
        img_key = str(image_path)
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
                if meta.get("image_path") != img_key:
                    return False  # imagem diferente — reprocessar
            except: pass
        else:
            return False  # sem sidecar = gerado antes deste sistema, reprocessar
    return True

def resolve_project_root(pairs) -> Path:
    """Determina a pasta raiz do projeto a partir dos áudios fornecidos."""
    for pair in pairs:
        audio = pair[0]
        if isinstance(audio, (tuple, list)):
            audio = audio[0]
        if audio and os.path.exists(audio):
            return Path(audio).parent
    return Path.cwd()



def _smoothstep_tensor(t: torch.Tensor, better: bool = True) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    if better: return t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    return t * t * (3.0 - 2.0 * t)

_BLUR_KERNEL_CACHE: dict = {}

def _get_blur_kernel(radius: int, channels: int, device_str: str) -> torch.Tensor:
    """
    Retorna kernel gaussiano cacheado por (radius, channels, device).
    Criado uma vez por processo — reutilizado em todas as cenas.
    """
    key = (radius, channels, device_str)
    if key not in _BLUR_KERNEL_CACHE:
        k_size = radius * 2 + 1
        sigma  = radius / 2.0
        x      = torch.arange(k_size, device=device_str).float() - radius
        k1d    = torch.exp(-x**2 / (2 * sigma**2))
        k1d    = k1d / k1d.sum()
        k2d    = (k1d.view(1, 1, k_size, 1) * k1d.view(1, 1, 1, k_size))
        k2d    = k2d.expand(channels, 1, k_size, k_size).contiguous()
        _BLUR_KERNEL_CACHE[key] = k2d
    return _BLUR_KERNEL_CACHE[key]

# ---------------------------------------------------------------------------
# Worker: Renderização de Cena Individual
# ---------------------------------------------------------------------------
def _python_render_clip(
    image_paths: Union[str, Tuple[str, str]],
    audio_path:  Union[str, Tuple[str, str]],
    effect:      str,
    duration:    float,
    output_mp4:  str,
    fps:         int,
    encoder:     str,
    log_fn:      Optional[Callable] = None,
    transition_mode:  str   = "none",
    transition_time:  float = 0.0,
    scene_idx:        int   = 0,
    config:           dict  = None
) -> Tuple[bool, float]:
    _ensure_warmup()
    t0_render = time.monotonic()
    
    if not _TORCH_AVAILABLE: return False, 0.0
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        frames = int(duration * fps)
        if frames <= 0: return False, 0.0

        is_split = isinstance(image_paths, (tuple, list)) and len(image_paths) == 2
        
        # --- Helpers de Assets ---
        def load_asset(path, max_w, max_h):
            cache_key = (path, max_w, max_h, str(device))
            global _IMAGE_TENSOR_CACHE
            if cache_key in _IMAGE_TENSOR_CACHE:
                return _IMAGE_TENSOR_CACHE[cache_key]

            with Image.open(path) as im:

                img = im.convert("RGBA")
            orig_w, orig_h = img.width, img.height

            # Pré-resize grosseiro se for gigante
            MAX_UPLOAD_SIZE = 4096
            if max(orig_w, orig_h) > MAX_UPLOAD_SIZE * 2:
                pre_scale = MAX_UPLOAD_SIZE * 2 / max(orig_w, orig_h)
                pre_w = int(orig_w * pre_scale)
                pre_h = int(orig_h * pre_scale)
                img = img.resize((pre_w, pre_h), Image.Resampling.BILINEAR)
                orig_w, orig_h = pre_w, pre_h

            arr_orig = np.array(img)
            t_orig   = torch.from_numpy(arr_orig).permute(2,0,1).unsqueeze(0).float().to(device)

            # Resize na GPU
            rat   = min(max_w / orig_w, max_h / orig_h)
            new_w = int(orig_w * rat)
            new_h = int(orig_h * rat)
            t_img = F.interpolate(
                t_orig, size=(new_h, new_w),
                mode='bicubic', align_corners=False, antialias=True
            ).clamp(0, 255)

            pad_size = 200
            shadow_off = 28
            base_w = new_w + pad_size * 2
            base_h = new_h + pad_size * 2

            canvas = torch.zeros(1, 4, base_h, base_w, device=device)
            sy = pad_size + shadow_off
            sx = pad_size + shadow_off
            canvas[0, 3, sy:sy+new_h, sx:sx+new_w] = 255.0

            k_shadow = _get_blur_kernel(radius=28, channels=4, device_str=str(device))
            canvas   = F.conv2d(canvas, k_shadow, padding=28, groups=4).clamp(0, 255)

            py = pad_size
            px = pad_size
            alpha_img = t_img[:, 3:4, :, :] / 255.0
            canvas[:, :, py:py+new_h, px:px+new_w].mul_(1.0 - alpha_img).add_(
                t_img * alpha_img
            )

            res = (canvas, base_w, base_h)
            if len(_IMAGE_TENSOR_CACHE) >= _IMAGE_CACHE_MAX:
                oldest_key = next(iter(_IMAGE_TENSOR_CACHE))
                del _IMAGE_TENSOR_CACHE[oldest_key]
            _IMAGE_TENSOR_CACHE[cache_key] = res
            return res

        def paste_to_bg(bg, fg, x0, y0):
            hb, wb = bg.shape[2], bg.shape[3]
            hf, wf = fg.shape[2], fg.shape[3]
            x, y = int(x0), int(y0)
            y1, y2 = max(0, y), min(hb, y + hf)
            x1, x2 = max(0, x), min(wb, x + wf)
            if y2 <= y1 or x2 <= x1: return
            fy1, fx1 = y1 - y, x1 - x
            fy2, fx2 = fy1 + (y2 - y1), fx1 + (x2 - x1)
            alpha = fg[:, 3:4, fy1:fy2, fx1:fx2] / 255.0
            bg[:, :3, y1:y2, x1:x2].mul_(1.0 - alpha).add_(fg[:, :3, fy1:fy2, fx1:fx2] * alpha)

        # --- Carregamento ---
        bg_path = image_paths[0] if is_split else image_paths
        with Image.open(bg_path).convert("RGB") as im:
            orig_w, orig_h = im.width, im.height
            MAX_UPLOAD_SIZE = 4096
            if max(orig_w, orig_h) > MAX_UPLOAD_SIZE * 2:
                pre_scale = MAX_UPLOAD_SIZE * 2 / max(orig_w, orig_h)
                im = im.resize((int(orig_w * pre_scale), int(orig_h * pre_scale)), Image.Resampling.BILINEAR)

            rat = max(1920/im.width, 1080/im.height)
            nw, nh = int(im.width*rat), int(im.height*rat)
            bg_f = im.resize((nw, nh), Image.Resampling.LANCZOS)
            bx, by = (nw-1920)//2, (nh-1080)//2
            bg_c = bg_f.crop((bx, by, bx+1920, by+1080))
            bg_np = np.array(bg_c)

        bg_t_base = torch.from_numpy(bg_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        k_bg = _get_blur_kernel(radius=25, channels=3, device_str=str(device))
        bg_t_base = F.conv2d(bg_t_base, k_bg, padding=25, groups=3).clamp(0, 255)

        if is_split:
            fg1_t, w1, h1 = load_asset(image_paths[0], 850, 900)
            fg2_t, w2, h2 = load_asset(image_paths[1], 850, 900)
            cx1, cy1 = (960 - w1)//2, (1080 - h1)//2
            cx2, cy2 = 960 + (960 - w2)//2, (1080 - h2)//2
        else:
            with Image.open(image_paths) as p: pw, ph = p.size
            max_w, max_h = (1100, 1020) if ph > pw*1.2 else (1400, 900)
            fg_t, w, h = load_asset(image_paths, max_w, max_h)
            cx, cy = (1920 - w)//2, (1080 - h)//2

        if log_fn:
            log_fn(f"[GPU] Cena #{scene_idx+1}: BG blur GPU ✓ | Shadow blur GPU ✓ | Resize GPU ✓")

        # --- Valida existência dos arquivos antes de renderizar ---
        _audio_files = list(audio_path) if isinstance(audio_path, (tuple, list)) else [audio_path]
        _image_files = list(image_paths) if isinstance(image_paths, (tuple, list)) else [image_paths]
        for _f in _audio_files + _image_files:
            if not os.path.exists(_f):
                if log_fn: log_fn(f"❌ Cena #{scene_idx+1}: Arquivo não encontrado — {_f}")
                return False, 0.0

        # --- Encoder Config (Blackwell) ---
        if encoder == "h264_nvenc":
            vcodec = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "20",
                      "-b:v", "0", "-maxrate", "18M", "-bufsize", "36M", "-multipass", "fullres",
                      "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "20", "-profile:v", "high"]
        else:
            vcodec = ["-c:v", "libx264", "-preset", "faster", "-crf", "20"]

        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "1920x1080", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-"]
        if is_split:
            cmd.extend(["-i", audio_path[0], "-i", audio_path[1], "-filter_complex", "[1:a][2:a]concat=n=2:v=0:a=1[a_out]", "-map", "0:v", "-map", "[a_out]"])
        else:
            cmd.extend(["-i", audio_path, "-map", "0:v", "-map", "1:a"])
        
        # Filtros Visuais
        cfg_v = (config or {}).get("production", {}).get("video", {})
        vf = []
        if cfg_v.get("color_grading", True):
            vf.append("eq=contrast=1.05:brightness=0.02:saturation=1.1")
        if cfg_v.get("vibrance", False):
            vf.append("eq=saturation=1.4:gamma_r=1.02:gamma_b=0.98")
        if cfg_v.get("sharpen", True):
            vf.append("unsharp=3:3:0.5:3:3:0")
        if cfg_v.get("denoise", False):
            vf.append("hqdn3d=4:3:6:4")
        if cfg_v.get("film_grain", False):
            vf.append("noise=alls=15:allf=t")
        if vf: cmd.extend(["-vf", ",".join(vf)])
        
        cmd.extend([*vcodec, "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", output_mp4])

        # --- Loop de Renderização com Thread de Escrita Assíncrona ---
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        
        write_queue = queue.Queue(maxsize=4)
        def stdin_writer():
            try:
                while True:
                    data = write_queue.get()
                    if data is None:
                        write_queue.task_done()
                        break
                    CHUNK = 4 * 1024 * 1024
                    for offset in range(0, len(data), CHUNK):
                        proc.stdin.write(data[offset:offset + CHUNK])
                    write_queue.task_done()
            except Exception as e:
                if log_fn: log_fn(f"Erro no writer_thread: {e}")
            finally:
                try: proc.stdin.close()
                except: pass

        writer_thread = threading.Thread(target=stdin_writer, daemon=True)
        writer_thread.start()

        try:
            t_all = torch.linspace(0, 1, frames, device=device)
            e_all = _smoothstep_tensor(t_all, better=cfg_v.get("better_easing", True))
            fade_all = torch.ones(frames, device=device)
            f_len = int(transition_time * fps)
            if f_len > frames // 2: f_len = frames // 2
            if f_len > 0 and transition_mode not in ("none", "", None):
                if scene_idx > 0: fade_all[:f_len] = torch.linspace(0, 1, f_len, device=device)
                fade_all[-f_len:] = torch.linspace(1, 0, f_len, device=device)

            if not is_split:
                z_s, z_e, dx_s, dx_e, dy_s, dy_e = 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
                dist = h * 0.08
                if effect == "zoom_in": z_e = 1.10
                elif effect == "zoom_out": z_s = 1.10
                elif effect == "pan_up": dy_s, dy_e, z_s, z_e = dist, -dist, 1.08, 1.08
                elif effect == "pan_down": dy_s, dy_e, z_s, z_e = -dist, dist, 1.08, 1.08
                elif effect == "pan_left": dx_s, dx_e, z_s, z_e = dist, -dist, 1.08, 1.08
                elif effect == "pan_right": dx_s, dx_e, z_s, z_e = -dist, dist, 1.08, 1.08
                z_vals, dx_vals, dy_vals = z_s + (z_e-z_s)*e_all, dx_s + (dx_e-dx_s)*e_all, dy_s + (dy_e-dy_s)*e_all
            else:
                split_effects = ["zoom_in", "pan_left", "zoom_out", "pan_right"]
                split_eff = split_effects[scene_idx % 4]
                z1_base = 1.08 if "zoom" in split_eff else 1.05
                z1_vals = z1_base + (1.10 - z1_base) * e_all if split_eff == "zoom_in" else z1_base - (z1_base - 1.0) * e_all
                dy2_base = h2 * 0.06
                dy2_vals = dy2_base * (1 - 2 * e_all) if "pan" in split_eff else torch.zeros_like(e_all)

            frame_buf = torch.empty_like(bg_t_base)
            batch_frames_list = []
            for i in range(frames):
                frame_buf.copy_(bg_t_base)
                f_bg = frame_buf
                
                if is_split:
                    a1 = _render_anim_impl(fg1_t, w1, h1, z1_vals[i].item(), 0, 0)
                    paste_to_bg(f_bg, a1, cx1, cy1)
                    a2 = _render_anim_impl(fg2_t, w2, h2, 1.0, 0, dy2_vals[i].item())
                    paste_to_bg(f_bg, a2, cx2, cy2)
                else:
                    a = _render_anim_impl(fg_t, w, h, z_vals[i].item(), dx_vals[i].item(), dy_vals[i].item())
                    paste_to_bg(f_bg, a, cx, cy)
                
                if transition_mode in ("none", "", None):
                    f_rgb = f_bg.clamp(0, 255)
                else:
                    f_val = fade_all[i].item()
                    if transition_mode == "blur" and f_val < 1.0:
                        r = int((1-f_val)*15 + 0.5)
                        if r > 0: f_bg = F.conv2d(f_bg.clamp(0, 255), _get_blur_kernel(r, 3, str(device)), padding=r, groups=3)
                        f_rgb = f_bg.clamp(0, 255) * f_val
                    else:
                        f_rgb = f_bg.clamp(0, 255) * f_val
                
                batch_frames_list.append(f_rgb.squeeze(0).permute(1, 2, 0).to(torch.uint8))
                
                if len(batch_frames_list) == BATCH_SIZE or i == frames - 1:
                    batch_cpu = torch.stack(batch_frames_list).cpu()
                    data = batch_cpu.numpy().tobytes()
                    write_queue.put(data)
                    batch_frames_list.clear()

            write_queue.put(None)
            writer_thread.join()
        except Exception as render_err:
            # [FIX] Cancela o processo FFmpeg e drena a fila para desbloquear o writer_thread
            if log_fn: log_fn(f"❌ Erro de renderização Cena #{scene_idx+1}: {render_err}")
            try: write_queue.put(None, block=False)
            except: pass
            writer_thread.join(timeout=3)
            try: proc.terminate(); proc.wait(timeout=5)
            except: pass
            return False, 0.0
        
        proc.wait()
        success = proc.returncode == 0
        elapsed = time.monotonic() - t0_render

        # Salva metadados da cena apenas em sucesso
        if success:
            try:
                sidecar_path = Path(output_mp4).with_suffix(".json")
                img_key = image_paths[0] if isinstance(image_paths, (tuple,list)) else image_paths
                sidecar_path.write_text(json.dumps({"image_path": str(img_key)}))
            except: pass
        
        return success, elapsed
    except Exception as e:
        if log_fn: log_fn(f"Erro Cena #{scene_idx+1}: {e}")
        return False, 0.0

# ---------------------------------------------------------------------------
# Pipeline Principal
# ---------------------------------------------------------------------------
class VideoPipeline(QObject):
    progress = Signal(int, int)
    video_progress = Signal(int, int)
    log_message = Signal(str)
    finished = Signal(bool, str)
    video_scene_done = Signal(int, int)
    video_complete = Signal(str, float, float)

    def __init__(self, pairs, output_path, effect_mode="auto", layout_mode="single",
                 transition_mode="none", transition_time=0.0, bg_music_path="", bg_music_volume=10, config=None, parent=None):
        super().__init__(parent)
        self.pairs, self.output_path = pairs, output_path
        self.effect_mode, self.layout_mode = effect_mode, layout_mode
        self.transition_mode, self.transition_time = transition_mode, transition_time
        self.bg_music_path, self.bg_music_volume = bg_music_path, bg_music_volume
        self.config, self._cancelled = config or {}, False

    def cancel(self): self._cancelled = True

    def run(self):
        if not self.pairs: return self.finished.emit(False, "Sem pares áudio/imagem.")
        if not _ffmpeg_ok(): return self.finished.emit(False, "FFmpeg não encontrado.")
        
        layout_labels = {
            "single": "Uma Imagem por Cena",
            "split": "Duas Imagens por Cena",
            "sequential": "Alternar Sequencial (3-5)",
            "random": "Alternar Aleatório (70/30)",
        }
        self.log_message.emit(
            f"🎬 Layout: {layout_labels.get(self.layout_mode, self.layout_mode)} | "
            f"{sum(1 for p in self.pairs if not isinstance(p[0], tuple))} single | "
            f"{sum(1 for p in self.pairs if isinstance(p[0], tuple))} split"
        )
        
        self.log_message.emit(f"🚀 Iniciando Renderização Otimizada (Batch Writing 60FPS)\nDestino: {self.output_path}\n")
        
        # Determina a pasta de saída e de trabalho (cenas temporárias)
        # Se for um path absoluto, respeitamos. Se for relativo, organizamos em /videos
        target_path = Path(self.output_path)
        project_root = resolve_project_root(self.pairs)
        
        if not target_path.is_absolute():
            base_root = project_root.parent if project_root.name.lower() == "audios" else project_root
            videos_dir = base_root / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = str(videos_dir / target_path.name)
        else:
            videos_dir = target_path.parent
            videos_dir.mkdir(parents=True, exist_ok=True)

        # Pasta de cenas (temporárias) fica sempre junta da saída final
        scenes_dir = videos_dir / f"{Path(self.output_path).stem}_cenas"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_message.emit(f"📁 Pasta do projeto: {project_root}")
        self.log_message.emit(f"🎬 Destino final: {self.output_path}")

        existing = [f for f in scenes_dir.glob("scene_*.mp4") if f.exists()]
        if existing:
            self.log_message.emit(f"♻️  {len(existing)} cena(s) encontrada(s) — serão validadas antes de renderizar")
        else:
            self.log_message.emit("🆕 Nenhuma cena prévia encontrada — renderização completa")

        encoder, start_time = _get_best_encoder(), time.time()

        clip_tasks = []
        base_effs = []
        last_eff = None
        for _ in range(len(self.pairs)):
            available = [e for e in EFFECTS if e != last_eff]
            eff = random.choice(available)
            base_effs.append(eff)
            last_eff = eff
        
        for i, pair in enumerate(self.pairs):
            is_split = isinstance(pair[0], tuple)
            if is_split:
                a_p, i_p = pair[0], pair[1]
                dur = _audio_duration(a_p[0]) + _audio_duration(a_p[1])
            else:
                a_p, i_p = pair[0], pair[1]
                dur = _audio_duration(a_p)
            
            if dur <= 0: continue
            
            if self.effect_mode != "auto":
                eff = self.effect_mode
            elif is_split:
                eff = "auto"  # split resolve internamente por scene_idx
            else:
                eff = base_effs[i]  # single: efeito do pool embaralhado
                
            out = str(scenes_dir / f"scene_{i+1:03d}.mp4")
            clip_tasks.append((i+1, a_p, i_p, eff, dur, out))

        if not clip_tasks: return self.finished.emit(False, "Durações inválidas.")
        
        total, completed, frames_total = len(clip_tasks), 0, sum(int(t[4]*FPS) for t in clip_tasks)
        frames_done, concat_map = 0, {}

        # Resume System Check
        valid_tasks = []
        for i, t in enumerate(clip_tasks):
            idx_1, a_p, i_p, eff, dur, out = t
            if _validate_existing_clip(out, a_p, i_p):
                self.log_message.emit(f"  ⏭ Cena #{idx_1:02d} (pos {i}) já existe e é válida — pulando")
                concat_map[i] = out
                completed += 1
                frames_done += int(dur * FPS)
                self.progress.emit(completed, total)
                self.video_scene_done.emit(completed, total)
            else:
                valid_tasks.append((i, t))

        avg_dur = sum(t[4] for t in clip_tasks) / len(clip_tasks) if clip_tasks else 6.0
        if encoder == "h264_nvenc":
            # Cenas longas (>10s) consomem mais VRAM: usa 1 worker para evitar throttle
            max_workers = 1 if avg_dur > 10.0 else 2
        else:
            max_workers = os.cpu_count() or 4

        self.log_message.emit(
            f"⚙️  Workers: {max_workers} | Duração média por cena: {avg_dur:.1f}s | Encoder: {encoder}"
        )
        _log_q = queue.Queue()
        def _flush_log_queue():
            while not _log_q.empty(): self.log_message.emit(_log_q.get_nowait())

        _eta_history: list = []        # histórico de fps pontuais por cena
        _eta_start_time = time.time()  # referência local para o ETA
        processed = 0  # conta todas as cenas tentadas (sucesso + falha)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_to_task = {executor.submit(_python_render_clip, t[2], t[1], t[3], t[4], t[5], FPS, encoder, _log_q.put, self.transition_mode, self.transition_time, i, self.config): (i, t) for i, t in valid_tasks}
            
            for fut in concurrent.futures.as_completed(fut_to_task):
                if self._cancelled:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return self.finished.emit(False, "Cancelado.")
                
                idx, t = fut_to_task[fut]
                try: success, r_time = fut.result()
                except: success, r_time = False, 0.0
                
                elapsed = time.time() - _eta_start_time

                # Só calcula após 3s reais de dados — evita explosão no início paralelo
                if elapsed >= 3.0 and (frames_done > 0 or processed > 0):
                    fps_instant = int(t[4] * FPS) / r_time if r_time > 0 else 1
                    _eta_history.append(fps_instant)
                    if len(_eta_history) > 8:          # janela dos últimos 8 resultados
                        _eta_history.pop(0)
                    fps_smooth  = sum(_eta_history) / len(_eta_history)
                    eta_sec     = int((frames_total - frames_done) / fps_smooth)
                    if eta_sec <= 0:
                        eta_s = "finalizando..."
                    elif eta_sec >= 3600:
                        eta_s = f"{eta_sec//3600}h {(eta_sec%3600)//60}m"
                    elif eta_sec >= 60:
                        eta_s = f"{eta_sec//60}m {eta_sec%60}s"
                    else:
                        eta_s = f"{eta_sec}s"
                else:
                    eta_s = "calculando..."
                
                if success and os.path.exists(t[5]):
                    speed = t[4]/r_time if r_time > 0 else 0
                    completed += 1
                    frames_done += int(t[4] * FPS)
                    self.log_message.emit(f"  ✓ Cena #{t[0]:02d} | {t[4]:.1f}s | {int(t[4]*FPS)} frames | render: {r_time:.1f}s | {speed:.1f}x realtime | ETA: {eta_s}")
                    concat_map[idx] = t[5]
                else: self.log_message.emit(f"  ✗ Falha Cena #{t[0]:02d} | ETA: {eta_s}")
                
                processed += 1
                _flush_log_queue()
                self.progress.emit(processed + len(clip_tasks) - len(valid_tasks), total)
                self.video_scene_done.emit(processed + len(clip_tasks) - len(valid_tasks), total)

        _flush_log_queue()
        
        # Concatenação e Finalização
        all_indices = set(range(len(clip_tasks)))
        rendered    = set(concat_map.keys())
        failed      = sorted(all_indices - rendered)

        if failed:
            self.log_message.emit(
                f"⚠️  {len(failed)} cena(s) falharam e serão omitidas do vídeo final: "
                f"{[clip_tasks[i][0] for i in failed]}"
            )

        c_list = [concat_map[k] for k in sorted(concat_map)]
        if not c_list: return self.finished.emit(False, "Sem cenas geradas.")
        
        self.log_message.emit("\nAgrupando vídeo final...")
        c_txt = scenes_dir / "concat.txt"
        with open(c_txt, "w", encoding="utf-8") as f:
            for p in c_list: f.write(f"file '{Path(p).name}'\n")
            
        out_p = Path(self.output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        self.log_message.emit(f"🔗 Concatenando {len(c_list)} cenas...")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(c_txt), "-c", "copy", str(out_p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not out_p.exists() or out_p.stat().st_size < 1024:
            self.finished.emit(False, "Falha na concatenação — arquivo final inválido ou vazio.")
            return
        self.log_message.emit(f"✓ Concatenação validada: {out_p.stat().st_size / (1024*1024):.1f} MB")
        self.log_message.emit("✓ Concatenação concluída!")
        
        # BGM Auto-ducking (opcional via sidechain)
        if self.bg_music_path and out_p.exists():
            self.log_message.emit("🎵 Adicionando BGM + Auto-ducking (pode demorar)...")
            tmp = out_p.with_name(f"{out_p.stem}_tmp.mp4")
            vol = max(0.01, min(self.bg_music_volume / 100.0, 2.0))
            if vol <= 0.05:
                self.log_message.emit("⚠️  Volume do BGM muito baixo — considere aumentar acima de 5%")
            use_ducking = (self.config or {}).get("production", {}).get("auto_ducking", False)

            if use_ducking:
                fc = (f"[0:a]asplit=2[v][vs];"
                      f"[1:a]volume={vol}[bg];"
                      f"[bg][vs]sidechaincompress=threshold=-25dB:ratio=3:attack=200:release=1000[bgd];"
                      f"[v][bgd]amix=inputs=2:duration=first[a]")
            else:
                fc = f"[1:a]volume={vol}[bg];[0:a][bg]amix=inputs=2:duration=first[a]"

            subprocess.run(["ffmpeg", "-y", "-i", str(out_p), "-stream_loop", "-1", "-i", self.bg_music_path, "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", str(tmp)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if tmp.exists():
                self.log_message.emit("✓ BGM finalizado!")
                shutil.move(str(tmp), str(out_p))
                self.log_message.emit("✓ Arquivo final salvo!")

        total_time = time.time() - start_time
        total_dur = sum(t[4] for t in clip_tasks)
        speed = total_dur / total_time if total_time > 0 else 0
        mr, sr = divmod(int(total_time), 60)
        md, sd = divmod(int(total_dur), 60)

        self.log_message.emit("═"*42 + f"\n✓ Vídeo finalizado!\n   Cenas        : {completed}\n   Duração      : {md}m {sd}s\n   Render       : {mr}m {sr}s\n   Velocidade   : {speed:.2f}x realtime\n   Encoder      : {encoder}\n   Saída        : {out_p}\n" + "═"*42)
        self.video_complete.emit(str(out_p), total_time, speed)
        self.finished.emit(True, "Sucesso")