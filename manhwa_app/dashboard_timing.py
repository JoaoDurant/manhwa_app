# manhwa_app/dashboard_timing.py
#
# DashboardTiming — Three-Clock Engine for the Macro Dashboard
# ============================================================
#
# Encapsula a lógica dos 3 relógios do painel de controle:
#
#   Clock 1 (Azul)  — 🎙️ Audio Generation
#     Conta desde o início do parágrafo atual; reseta a cada novo parágrafo.
#
#   Clock 2 (Âmbar) — 📁 Current Job
#     Conta desde o início do job; ETA rolling baseado em avg_s_per_para.
#
#   Clock 3 (Roxo)  — 📦 Full Queue
#     Conta desde que a fila começou; ETA baseado em média histórica de jobs.
#
# Todos os cálculos e a QTimer rodam inteiramente na main thread.
# NÃO há acesso a GPU / tensores aqui — apenas aritmética de time.time().
#
# Integração:
#   timing = DashboardTiming()
#   timing.tick.connect(lambda p, j, je, q, qe: ...)
#   timing.start_queue()
#   # Depois conecte on_para_started, on_para_done, on_job_started, on_job_done

import time
from PySide6.QtCore import QObject, QTimer, Signal


class DashboardTiming(QObject):
    """
    Emite tick() a cada 500ms com os valores dos 3 relógios e seus ETAs.

    Sinal tick(para_clock, job_clock, job_eta, queue_clock, queue_eta):
        para_clock  — str "HH:MM:SS" do relógio 1 (parágrafo atual)
        job_clock   — str "HH:MM:SS" do relógio 2 (job atual)
        job_eta     — str "HH:MM:SS" ou "calculando..." do ETA do job
        queue_clock — str "HH:MM:SS" do relógio 3 (fila total)
        queue_eta   — str "HH:MM:SS" ou "calculando..." do ETA da fila
    """
    tick = Signal(str, str, str, str, str)

    _PLACEHOLDER = "--:--:--"
    _CALC_MSG    = "calculando..."

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Clock 1 — Parágrafo Atual ────────────────────────────────────
        self._para_start:   float = 0.0
        self._para_idx:     int   = 0
        self._para_total:   int   = 0

        # ── Clock 2 — Job Atual ──────────────────────────────────────────
        self._job_start:     float       = 0.0
        self._job_idx:       int         = 0      # 0-based
        self._total_jobs:    int         = 0
        self._para_times:    list[float] = []     # elapsed_s por parágrafo feito
        self._total_paras:   int         = 0      # total do job atual
        self._last_video_dur: float      = 60.0   # heurística inicial; atualizada em video_complete

        # ── Clock 3 — Fila ──────────────────────────────────────────────
        self._queue_start:       float       = 0.0
        self._job_elapsed_list:  list[float] = []   # elapsed de cada job completo
        self._remaining_jobs:    int         = 0
        self._queue_running:     bool        = False

        # ─── QTimer 500ms ────────────────────────────────────────────────
        # Roda 100% na main thread. Sem acesso a GPU.
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._on_tick)

    # ── API Pública — Ciclo de Vida ─────────────────────────────────────

    def start_queue(self, total_jobs: int = 0):
        """Chamado quando o Macro Engine inicia a fila."""
        self._queue_start     = time.monotonic()
        self._queue_running   = True
        self._total_jobs      = total_jobs
        self._remaining_jobs  = total_jobs
        self._job_elapsed_list.clear()
        self._timer.start()

    def stop_queue(self):
        """Chamado quando a fila termina (all_done / queue_complete)."""
        self._queue_running = False
        self._timer.stop()
        # Emite estado final congelado
        self._emit_snapshot()

    def on_job_started(self, job_id: str, job_idx: int, total_jobs: int):
        """
        Chamado quando MacroCoordinator.job_started é emitido.
        job_idx é 0-based; reseta Clock 2 e stats de parágrafo.
        """
        self._job_start    = time.monotonic()
        self._job_idx      = job_idx
        self._total_jobs   = total_jobs
        self._remaining_jobs = total_jobs - job_idx
        self._para_times.clear()
        self._total_paras  = 0
        # Reseta para placeholder até o primeiro parágrafo
        self._para_start   = 0.0
        self._para_idx     = 0

    def on_job_done(self, elapsed_s: float):
        """
        Chamado quando MacroCoordinator.job_finished é emitido (success ou fail).
        elapsed_s é o tempo total do job.
        """
        if elapsed_s > 0:
            self._job_elapsed_list.append(elapsed_s)
        if self._remaining_jobs > 0:
            self._remaining_jobs -= 1

    def on_para_started(self, idx: int, total: int):
        """
        Chamado quando AudioPipeline.paragraph_started é emitido.
        Reseta Clock 1 e salva ctx do parágrafo atual.
        """
        self._para_start = time.monotonic()
        self._para_idx   = idx
        self._para_total = total
        self._total_paras = total

    def on_para_done(self, idx: int, total: int, elapsed_s: float):
        """
        Chamado quando AudioPipeline.paragraph_done_stats é emitido.
        Acumula o elapsed para cálculo de ETA.
        """
        if elapsed_s > 0:
            self._para_times.append(elapsed_s)
        self._para_total = total

    def on_video_complete(self, duration_s: float):
        """
        Atualiza a heurística de duração de vídeo para o ETA de job.
        duration_s = duração do vídeo final em segundos.
        """
        if duration_s > 0:
            self._last_video_dur = duration_s

    # ── Timer Callback ─────────────────────────────────────────────────

    def _on_tick(self):
        self._emit_snapshot()

    def _emit_snapshot(self):
        now = time.monotonic()

        # Clock 1 — Parágrafo
        if self._para_start > 0:
            para_s = now - self._para_start
        else:
            para_s = -1.0

        # Clock 2 — Job
        if self._job_start > 0:
            job_s = now - self._job_start
        else:
            job_s = -1.0

        # Clock 3 — Fila
        if self._queue_start > 0 and self._queue_running:
            queue_s = now - self._queue_start
        elif self._queue_start > 0:
            # Fila encerrada — mostra elapsed total (congelado)
            queue_s = now - self._queue_start
        else:
            queue_s = -1.0

        para_str  = self._fmt(para_s)  if para_s  >= 0 else self._PLACEHOLDER
        job_str   = self._fmt(job_s)   if job_s   >= 0 else self._PLACEHOLDER
        queue_str = self._fmt(queue_s) if queue_s >= 0 else self._PLACEHOLDER

        job_eta   = self._compute_job_eta()
        queue_eta = self._compute_queue_eta()

        self.tick.emit(para_str, job_str, job_eta, queue_str, queue_eta)

    # ── ETA Calculations ───────────────────────────────────────────────

    def _compute_job_eta(self) -> str:
        """
        ETA para o job atual (Clock 2 sub-label).

        Fórmula (exata do spec):
            avg_s_per_para = total_elapsed / paras_done
            audio_remaining = avg_s_per_para * (total - done)
            video_eta = last_video_duration_s * 0.15
            job_eta = max(0, audio_remaining + video_eta)

        Mostra "calculando..." até 3 parágrafos concluídos.
        """
        n = len(self._para_times)
        if n < 3:
            return self._CALC_MSG

        avg     = sum(self._para_times) / n
        remaining = max(0, self._total_paras - n)
        audio_eta = avg * remaining
        video_eta = self._last_video_dur * 0.15
        total_eta = max(0.0, audio_eta + video_eta)
        return self._fmt(total_eta)

    def _compute_queue_eta(self) -> str:
        """
        ETA para a fila inteira (Clock 3 sub-label).

        Se jobs completos > 0: usa média dos elapsed de jobs completos.
        Se nenhum completo: usa velocidade do job atual (para_times).
        Mostra "calculando..." até ter dados suficientes.
        """
        remaining = max(0, self._remaining_jobs)
        if remaining == 0 and self._queue_running:
            return self._fmt(0.0)

        # Caso 1: há jobs completos — usa média histórica
        if self._job_elapsed_list:
            avg_job = sum(self._job_elapsed_list) / len(self._job_elapsed_list)
            return self._fmt(max(0.0, avg_job * remaining))

        # Caso 2: nenhum job completo — projeta a partir do job atual
        n = len(self._para_times)
        if n < 3 or self._total_paras == 0:
            return self._CALC_MSG

        avg_para = sum(self._para_times) / n
        # Projeta duração total do job atual, depois multiplica pelos restantes
        projected_job = avg_para * self._total_paras
        return self._fmt(max(0.0, projected_job * remaining))

    # ── Helpers ────────────────────────────────────────────────────────

    # ── Test & Internal Raw API ────────────────────────────────────────

    def start_paragraph(self, idx: int, total: int):
        self.on_para_started(idx, total)

    def get_paragraph_elapsed(self) -> float:
        if self._para_start > 0:
            return time.monotonic() - self._para_start
        return -1.0

    def start_job(self, total_paragraphs: int):
        self.on_job_started("test_job", 0, 1)
        self._total_paras = total_paragraphs

    def record_paragraph_complete(self, elapsed_s: float):
        if elapsed_s > 0:
            self._para_times.append(elapsed_s)

    def get_job_elapsed(self) -> float:
        if self._job_start > 0:
            return time.monotonic() - self._job_start
        return -1.0

    def get_job_eta(self) -> float | None:
        n = len(self._para_times)
        if n < 3:
            return None
        avg = sum(self._para_times) / n
        remaining = max(0, self._total_paras - n)
        audio_eta = avg * remaining
        video_eta = self._last_video_dur * 0.15
        return max(0.0, audio_eta + video_eta)

    def record_job_complete(self, elapsed_s: float):
        self.on_job_done(elapsed_s)

    def get_queue_eta(self, remaining_jobs: int) -> float | None:
        self._remaining_jobs = remaining_jobs
        if remaining_jobs == 0 and self._queue_running:
            return 0.0
        if self._job_elapsed_list:
            avg_job = sum(self._job_elapsed_list) / len(self._job_elapsed_list)
            return max(0.0, avg_job * remaining_jobs)
        n = len(self._para_times)
        if n < 3 or self._total_paras == 0:
            return None
        avg_para = sum(self._para_times) / n
        projected_job = avg_para * self._total_paras
        return max(0.0, projected_job * remaining_jobs)

    @staticmethod
    def _fmt(seconds: float) -> str:
        s = max(0.0, seconds)
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"
