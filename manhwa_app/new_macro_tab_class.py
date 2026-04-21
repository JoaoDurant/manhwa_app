import json
from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QSplitter, QScrollArea, QGroupBox, QFormLayout,
                               QLineEdit, QComboBox, QLabel, QTableWidget,
                               QHeaderView, QTextEdit, QFrame, QMessageBox,
                               QFileDialog, QProgressBar, QTableWidgetItem,
                               QApplication)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor

from manhwa_app.dashboard_timing import DashboardTiming
from manhwa_app.macro_core import MacroCoordinator, MacroJob

class MacroTab(QWidget):
    """Aba de Macro Geral para automação total do fluxo de trabalho."""
    
    macro_log = Signal(str)
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.coordinator = MacroCoordinator(main_window, parent=self)
        self._current_txt = ""
        self._img_dir = ""
        self._audio_dir = ""
        self._job_widgets = {} # job_id -> dict
        
        # Dashboard Analytics State
        self.timing = DashboardTiming(self)
        self._total_paras_done = 0
        self._total_retries = 0
        self._sim_sum = 0.0
        self._rms_sum = 0.0
        self._fastest_para = 9999.0
        self._slowest_para = 0.0

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        main_v = QVBoxLayout(self)
        main_v.setContentsMargins(15, 15, 15, 15)
        main_v.setSpacing(15)

        # Barra Superior: Controles e Bulk
        top_bar = QHBoxLayout()
        btn_bulk = QPushButton("📂 Importar Pasta de Roteiros (Auto)")
        btn_bulk.clicked.connect(self._bulk_import)
        
        btn_clear = QPushButton("🗑 Limpar Tudo")
        btn_clear.clicked.connect(self._clear_all)
        
        btn_save_q = QPushButton("💾 Salvar Fila")
        btn_save_q.clicked.connect(self._export_queue)
        
        btn_load_q = QPushButton("📂 Carregar Fila")
        btn_load_q.clicked.connect(self._import_queue)
        
        top_bar.addWidget(btn_bulk)
        top_bar.addStretch()
        top_bar.addWidget(btn_save_q)
        top_bar.addWidget(btn_load_q)
        top_bar.addWidget(btn_clear)
        main_v.addLayout(top_bar)

        # Splitter central: Adição vs Dashboard
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LADO ESQUERDO: Formulário de Adição (Intacto) ---
        add_scroll = QScrollArea()
        add_scroll.setWidgetResizable(True)
        add_scroll.setMinimumWidth(350)
        form_w = QWidget()
        form_v = QVBoxLayout(form_w)
        
        group_add = QGroupBox("➕ Nova Tarefa Macro")
        fl = QFormLayout(group_add)
        fl.setSpacing(10)
        
        self.edit_proj = QLineEdit("projeto_batch")
        fl.addRow("Nome/Pasta:", self.edit_proj)
        
        self.combo_workflow = QComboBox()
        self.combo_workflow.addItem("🔊 Apenas Áudio (TTS)", "audio")
        self.combo_workflow.addItem("🎬 Áudio + Vídeo (Full)", "audio_video")
        self.combo_workflow.addItem("🛠️ Apenas Vídeo (Edit Mode)", "video_edit")
        fl.addRow("Workflow:", self.combo_workflow)
        
        self.tts_container = QWidget()
        self.tts_fl = QFormLayout(self.tts_container)
        self.tts_fl.setContentsMargins(0, 0, 0, 0)
        
        self.btn_pick_txt = QPushButton("📄 Selecionar Script (.txt)")
        self.btn_pick_txt.clicked.connect(self._pick_txt)
        self.lbl_txt_val = QLabel("(nenhum)")
        self.lbl_txt_val.setStyleSheet("color:#666; font-size:11px;")
        self.tts_fl.addRow("Fonte (Script):", self.btn_pick_txt)
        self.tts_fl.addRow("", self.lbl_txt_val)

        self.combo_engine = QComboBox()
        self.combo_engine.addItem("Chatterbox", "chatterbox")
        self.combo_engine.addItem("Kokoro", "kokoro")
        self.combo_engine.currentTextChanged.connect(self._on_engine_changed)
        self.tts_fl.addRow("TTS Engine:", self.combo_engine)
        
        self.combo_model_type = QComboBox()
        self.tts_fl.addRow("Modelo/Tipo:", self.combo_model_type)
        
        self.combo_voice = QComboBox()
        self.combo_voice.setEditable(False)
        self.combo_voice.setPlaceholderText("Selecione a voz...")
        self.combo_voice.setMinimumHeight(32)
        
        self.btn_pick_voice = QPushButton("📁")
        self.btn_pick_voice.setFixedWidth(30)
        self.btn_pick_voice.clicked.connect(self._pick_voice)
        
        self.btn_refresh_v = QPushButton("🔄")
        self.btn_refresh_v.setFixedWidth(30)
        self.btn_refresh_v.clicked.connect(self._refresh_voices)
        
        voice_h = QHBoxLayout()
        voice_h.addWidget(self.combo_voice, 1)
        voice_h.addWidget(self.btn_refresh_v)
        voice_h.addWidget(self.btn_pick_voice)
        self.tts_fl.addRow("Voz / Clone:", voice_h)
        
        self.combo_lang = QComboBox()
        self.combo_lang.addItem("✨ Auto Detectar", "auto")
        for l in ["pt", "en", "es", "ja", "ko"]: self.combo_lang.addItem(l, l)
        self.tts_fl.addRow("Idioma:", self.combo_lang)
        
        fl.addRow(self.tts_container)

        self.video_container = QWidget()
        self.video_fl = QFormLayout(self.video_container)
        self.video_fl.setContentsMargins(0, 0, 0, 0)
        
        self.btn_pick_aud = QPushButton("🎵 Selecionar Pasta de Áudios")
        self.btn_pick_aud.clicked.connect(self._pick_aud_dir)
        self.lbl_aud_val = QLabel("(usar padrão do projeto)")
        self.lbl_aud_val.setStyleSheet("color:#666; font-size:11px;")
        self.video_fl.addRow("Áudio:", self.btn_pick_aud)
        self.video_fl.addRow("", self.lbl_aud_val)

        self.btn_pick_img = QPushButton("🖼️ Selecionar Pasta de Imagens (Opcional)")
        self.btn_pick_img.clicked.connect(self._pick_img_dir)
        self.lbl_img_val = QLabel("(usar padrão do projeto)")
        self.lbl_img_val.setStyleSheet("color:#666; font-size:11px;")
        self.video_fl.addRow("Imagens:", self.btn_pick_img)
        self.video_fl.addRow("", self.lbl_img_val)
        
        fl.addRow(self.video_container)
        
        self.combo_workflow.currentTextChanged.connect(self._on_workflow_changed)
        self._on_workflow_changed()
        self._on_engine_changed()
        
        btn_add = QPushButton("➕ Adicionar tarefa à Fila")
        btn_add.setObjectName("primary")
        btn_add.setMinimumHeight(40)
        btn_add.clicked.connect(self._add_single_job)
        fl.addRow(btn_add)
        
        form_v.addWidget(group_add)
        form_v.addStretch()
        add_scroll.setWidget(form_w)
        splitter.addWidget(add_scroll)

        # --- LADO DIREITO: DASHBOARD LIVE ---
        right_panel = QSplitter(Qt.Orientation.Vertical)
        
        # A. Timing Header Bar
        timing_w = QWidget()
        timing_h = QHBoxLayout(timing_w)
        timing_h.setContentsMargins(5, 5, 5, 5)
        # estilo glassmorphism sutil
        timing_w.setStyleSheet("QWidget { background: rgba(0,0,0,0.2); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); }")
        
        self.lbl_clock_audio = QLabel("🎙️ Audio: --:--:--")
        self.lbl_clock_audio.setStyleSheet("color: #00bcd4; font-size: 15px; font-weight: bold; padding: 5px;")
        self.lbl_clock_job = QLabel("📁 Job: --:--:--  ETA: --:--:--")
        self.lbl_clock_job.setStyleSheet("color: #ffb300; font-size: 15px; font-weight: bold; padding: 5px;")
        self.lbl_clock_queue = QLabel("📦 Queue: --:--:--  ETA: --:--:--")
        self.lbl_clock_queue.setStyleSheet("color: #ba68c8; font-size: 15px; font-weight: bold; padding: 5px;")
        
        timing_h.addWidget(self.lbl_clock_audio)
        timing_h.addStretch()
        timing_h.addWidget(self.lbl_clock_job)
        timing_h.addStretch()
        timing_h.addWidget(self.lbl_clock_queue)
        
        right_panel.addWidget(timing_w)
        
        # B. Fila de Jobs (Tabela)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["#", "Projeto", "Workflow", "Engine", "Status", "🎙️ Áudio", "🎬 Vídeo", "Elapsed", "Ação"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
        self.table.setColumnWidth(5, 120)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Fixed)
        self.table.setColumnWidth(6, 120)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget { background: transparent; border: none; }
            QTableWidget::item { padding: 5px; border-bottom: 1px solid rgba(255,255,255,0.05); }
        """)
        right_panel.addWidget(self.table)
        
        # C. Log e Stats Strip (Bottom Splitter)
        bottom_w = QWidget()
        bottom_h = QHBoxLayout(bottom_w)
        bottom_h.setContentsMargins(0, 0, 0, 0)
        
        # Live Log
        log_container = QWidget()
        log_v = QVBoxLayout(log_container)
        log_v.setContentsMargins(0, 0, 0, 0)
        
        log_tools = QHBoxLayout()
        self.combo_log_filter = QComboBox()
        self.combo_log_filter.addItems(["Todas as Mensagens", "Erros Apenas", "Avisos ou Erros"])
        log_tools.addWidget(self.combo_log_filter)
        log_tools.addStretch()
        btn_copy = QPushButton("📑 Copiar Log")
        btn_copy.clicked.connect(self._copy_log)
        log_tools.addWidget(btn_copy)
        
        self.log_html = QTextEdit()
        self.log_html.setReadOnly(True)
        self.log_html.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; background: rgba(0,0,0,0.3); border-radius: 4px;")
        
        log_v.addLayout(log_tools)
        log_v.addWidget(self.log_html)
        
        # Stats Strip
        self.stats_frame = QFrame()
        self.stats_frame.setFixedWidth(250)
        self.stats_frame.setStyleSheet("QFrame { background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); }")
        stats_v = QVBoxLayout(self.stats_frame)
        stats_v.addWidget(QLabel("<b>Estatísticas da Sessão</b>"))
        
        self.lbl_stats_paras = QLabel("Parágrafos: 0/0")
        self.lbl_stats_time = QLabel("Média Tempo: 0.0s")
        self.lbl_stats_extremes = QLabel("Rápido: - | Lento: -")
        self.lbl_stats_retries = QLabel("Retentativas: 0 (0%)")
        self.lbl_stats_quality = QLabel("Sim: 0.00 | RMS: 0.0")
        
        for lbl in [self.lbl_stats_paras, self.lbl_stats_time, self.lbl_stats_extremes, self.lbl_stats_retries, self.lbl_stats_quality]:
            lbl.setStyleSheet("color: #ccc; font-size: 12px;")
            stats_v.addWidget(lbl)
            
        stats_v.addStretch()
        
        bottom_h.addWidget(log_container, 3)
        bottom_h.addWidget(self.stats_frame, 1)
        right_panel.addWidget(bottom_w)
        
        # Distribuição de altura no right_panel: 5% header, 65% table, 30% log
        right_panel.setSizes([50, 400, 200])
        
        splitter.addWidget(right_panel)
        # Mais espaço para o lado direito
        splitter.setSizes([350, 850])
        main_v.addWidget(splitter, 1)

        # Barra de Ação Final
        self.btn_start = QPushButton("🚀 INICIAR VERIFICAÇÃO E MACRO")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.clicked.connect(self._start_macro)
        
        self.btn_pause = QPushButton("🛑 Parar Macro")
        self.btn_pause.clicked.connect(self.coordinator.stop)
        self.btn_pause.setEnabled(False)
        
        action_h = QHBoxLayout()
        action_h.addWidget(self.btn_start, 3)
        action_h.addWidget(self.btn_pause, 1)
        main_v.addLayout(action_h)

    def _setup_connections(self):
        # Timing
        self.timing.tick.connect(self._on_tick)
        
        # Coordinator (Macro)
        self.coordinator.queue_log.connect(self._on_queue_log)
        self.coordinator.job_started.connect(self._on_job_started)
        self.coordinator.job_progress.connect(self._update_audio_bar)
        self.coordinator.job_log.connect(self._on_job_log)
        self.coordinator.job_finished.connect(self._on_job_finished)
        self.coordinator.queue_complete.connect(self._on_queue_complete)
        self.coordinator.request_model_switch.connect(self.main_window.trigger_model_preload)

        # Precisamos conectar aos pipelines (Audio/Video) via a instância atual de AudioPipeline
        # O interceptador principal é injetado no job_started se o pipeline for recriado, 
        # mas como PySide6 permite conectar sinais de forma flexível, faremos uma proxy 
        # intercept function em `_start_job_workflow` no macro_core, ou interceptamos via os
        # eventos emitidos pelo pipeline atual.
        # Por segurança, vamos varrer pipeline a cada job_started via MainWindow.

    def _hook_pipelines(self):
        # Hook on current audio pipeline if exists and not hooked
        if self.coordinator._audio_pipeline and not hasattr(self.coordinator._audio_pipeline, "_dash_hooked"):
            pipe = self.coordinator._audio_pipeline
            pipe.paragraph_started.connect(self._on_para_started, Qt.ConnectionType.QueuedConnection)
            pipe.paragraph_done_stats.connect(self._on_para_done_stats, Qt.ConnectionType.QueuedConnection)
            pipe.paragraph_retry.connect(self._on_para_retry, Qt.ConnectionType.QueuedConnection)
            pipe._dash_hooked = True
            
        if self.coordinator._video_pipeline and not hasattr(self.coordinator._video_pipeline, "_dash_hooked"):
            pipe = self.coordinator._video_pipeline
            pipe.video_scene_done.connect(self._on_video_scene, Qt.ConnectionType.QueuedConnection)
            pipe.video_complete.connect(self._on_video_complete, Qt.ConnectionType.QueuedConnection)
            pipe._dash_hooked = True

    # -------------------------------------------------------------------------
    # PIPELINE HOOKS & TIMING
    # -------------------------------------------------------------------------
    @Slot(str, str, str, str, str)
    def _on_tick(self, para_c, job_c, job_e, queue_c, queue_e):
        self.lbl_clock_audio.setText(f"🎙️ Audio: {para_c}")
        self.lbl_clock_job.setText(f"📁 Job: {job_c}  ETA: {job_e}")
        self.lbl_clock_queue.setText(f"📦 Queue: {queue_c}  ETA: {queue_e}")
        
        # Atualiza a coluna de Tempo do job atual na tabela
        if self.coordinator._is_running and self.coordinator._current_idx >= 0:
            idx = self.coordinator._current_idx
            jid = self.coordinator.jobs[idx].id
            if jid in self._job_widgets:
                w = self._job_widgets[jid]
                if "lbl_time" in w:
                    w["lbl_time"].setText(job_c)

    @Slot(int, int, str)
    def _on_para_started(self, idx, total, preview):
        self.timing.on_para_started(idx, total)
        self.lbl_clock_audio.setText(f"🎙️ Audio: 00:00:00  (Para {idx}/{total})")
        self._append_dash_log("AUDIO", "INFO", f"Para {idx}/{total} iniciou: \"{preview}\"")

    @Slot(int, int, float, float, float, int)
    def _on_para_done_stats(self, idx, total, elapsed, sim, rms, attempts):
        self.timing.on_para_done(idx, total, elapsed)
        
        # Atualiza Stats Strip
        self._total_paras_done += 1
        self._sim_sum += sim
        self._rms_sum += rms
        self._total_retries += (attempts - 1)
        self._fastest_para = min(self._fastest_para, elapsed) if elapsed > 0 else self._fastest_para
        self._slowest_para = max(self._slowest_para, elapsed)
        
        avg_time = sum(self.timing._para_times) / len(self.timing._para_times) if self.timing._para_times else 0.0
        avg_sim = self._sim_sum / self._total_paras_done
        avg_rms = self._rms_sum / self._total_paras_done
        retry_pct = (self._total_retries / self._total_paras_done) * 100
        
        self.lbl_stats_paras.setText(f"Parágrafos: {idx}/{total}")
        self.lbl_stats_time.setText(f"Média Tempo: {avg_time:.1f}s")
        self.lbl_stats_extremes.setText(f"Rápido: {self._fastest_para:.1f}s | Lento: {self._slowest_para:.1f}s")
        self.lbl_stats_retries.setText(f"Retentativas: {self._total_retries} ({retry_pct:.1f}%)")
        self.lbl_stats_quality.setText(f"Sim: {avg_sim:.2f} | RMS: {avg_rms:.2f}")

        # Log simplificado e direto
        msg = f"Áudio {idx}/{total} finalizado em {elapsed:.1f}s"
        if attempts > 1: msg += f" (Retentativas: {attempts-1})"
        self._append_dash_log("AUDIO", "INFO", msg)

    @Slot(int, int, str)
    def _on_para_retry(self, idx, attempt, reason):
        self._append_dash_log("AUDIO", "WARN", f"Para {idx} retry {attempt}/3 ({reason})")

    @Slot(int, int)
    def _on_video_scene(self, current, total):
        if self.coordinator._is_running and self.coordinator._current_idx >= 0:
            jid = self.coordinator.jobs[self.coordinator._current_idx].id
            if jid in self._job_widgets:
                pb = self._job_widgets[jid]["vid_bar"]
                pb.setMaximum(total)
                pb.setValue(current)

    @Slot(str, float, float)
    def _on_video_complete(self, path, dur, el):
        self.timing.on_video_complete(dur)
        self._append_dash_log("VIDEO", "INFO", f"Composição concluída. Dura: {dur:.1f}s, Elapsed: {el:.1f}s")

    @Slot(str, int, int)
    def _on_job_started(self, jid, jidx, total):
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.timing.on_job_started(jid, jidx, total)
        self._hook_pipelines() # Prende os novos pipelines recém-criados
        
        # Atualiza Status na tabela
        if jid in self._job_widgets:
            self._job_widgets[jid]["status_lbl"].setText("🔵 Run")
            self._job_widgets[jid]["status_lbl"].setStyleSheet("color: #4fc3f7; font-weight: bold;")
            for col in range(self.table.columnCount()):
                item = self.table.item(self._job_widgets[jid]["row"], col)
                if item: item.setBackground(QColor(0, 50, 100, 50)) # Fundo azul leve

    @Slot(str, int, int)
    def _update_audio_bar(self, jid, current, total):
        if jid in self._job_widgets:
            pb = self._job_widgets[jid]["aud_bar"]
            if total > 0:
                pb.setMaximum(total)
                pb.setValue(current)
            elif total == -1:
                pb.setMaximum(100)
                pb.setValue(100)

    @Slot(str, bool, str)
    def _on_job_finished(self, jid, success, msg):
        # Apenas pega o snapshot do final
        elapsed_s = 0.0
        if self.timing._job_start > 0:
            import time
            elapsed_s = time.time() - self.timing._job_start
            self.timing.on_job_done(elapsed_s)
            
        if jid in self._job_widgets:
            st = "✅ Done" if success else "❌ Error"
            col = "#81c784" if success else "#e57373"
            self._job_widgets[jid]["status_lbl"].setText(st)
            self._job_widgets[jid]["status_lbl"].setStyleSheet(f"color: {col}; font-weight: bold;")
            if success:
                self._job_widgets[jid]["aud_bar"].setValue(self._job_widgets[jid]["aud_bar"].maximum())
            for c in range(self.table.columnCount()):
                item = self.table.item(self._job_widgets[jid]["row"], c)
                bg = QColor(0, 100, 0, 30) if success else QColor(100, 0, 0, 30)
                if item: item.setBackground(bg)
                
    @Slot(float)
    def _on_queue_complete(self, total_s):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.timing.stop_queue()
        QMessageBox.information(self, "Macro Finalizado", f"Fila processada com sucesso!\nTempo Gasto: {self.timing._fmt(total_s)}")

    # -------------------------------------------------------------------------
    # LOGGER COMPONENTE
    # -------------------------------------------------------------------------
    def _append_dash_log(self, stage, level, msg):
        import time
        ts = time.strftime("%H:%M:%S")
        
        # Filtros
        cfilter = self.combo_log_filter.currentText()
        if cfilter == "Erros Apenas" and level != "ERROR": return
        if cfilter == "Avisos ou Erros" and level not in ("ERROR", "WARN"): return
        
        color_stage = {"AUDIO": "#00bcd4", "VIDEO": "#ff4081", "MACRO": "#ba68c8"}.get(stage, "#ccc")
        color_lvl = {"INFO": "#eee", "WARN": "#ffd54f", "ERROR": "#ef5350"}.get(level, "#eee")
        
        if level == "INFO":
            entry = f"<span style='color:#888'>[{ts}]</span> <span style='color:{color_stage};'>•</span> <span style='color:#eee'>{msg}</span>"
        else:
            entry = f"<span style='color:#888'>[{ts}]</span> <span style='color:{color_lvl}; font-weight:bold;'>[{level}]</span> <span style='color:{color_lvl}'>{msg}</span>"
            
        self.log_html.append(entry)
        scrollbar = self.log_html.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _play_job(self, job):
        import os
        from pathlib import Path
        path = Path(job.output_root) / job.project_name
        full_audio = path / f"{job.project_name}_full_audio.wav"
        video_file = path / f"{job.project_name}_final.mp4"
        
        if video_file.exists():
            os.startfile(str(video_file))
        elif full_audio.exists():
            os.startfile(str(full_audio))
        elif path.exists():
            os.startfile(str(path))
        else:
            QMessageBox.information(self, "Play", f"Arquivos ainda não gerados para '\\n{job.project_name}'.")

    def _edit_job(self, job):
        self.edit_proj.setText(job.project_name)
        if job.workflow:
            idx_wf = self.combo_workflow.findData(job.workflow)
            if idx_wf >= 0: self.combo_workflow.setCurrentIndex(idx_wf)
            
        if job.txt_path:
            self._current_txt = job.txt_path
            self.lbl_txt_val.setText(Path(job.txt_path).name)
            
        if job.img_dir:
            self._img_dir = job.img_dir
            self.lbl_img_val.setText(Path(job.img_dir).name)
            
        if job.engine:
            idx_eng = self.combo_engine.findData(job.engine)
            if idx_eng >= 0: self.combo_engine.setCurrentIndex(idx_eng)
            self._on_engine_changed()
            
        if job.model_type:
            idx_md = self.combo_model_type.findData(job.model_type)
            if idx_md >= 0: self.combo_model_type.setCurrentIndex(idx_md)
            
        if job.voice:
            idx_vc = self.combo_voice.findData(job.voice)
            if idx_vc >= 0: 
                self.combo_voice.setCurrentIndex(idx_vc)
            else:
                self.combo_voice.addItem(f"👤 {Path(job.voice).name} (Externo)", job.voice)
                self.combo_voice.setCurrentIndex(self.combo_voice.count()-1)
                
        if job.lang:
            idx_l = self.combo_lang.findData(job.lang)
            if idx_l >= 0: self.combo_lang.setCurrentIndex(idx_l)
            
        self._remove_job(job.id)
        self.edit_proj.setFocus()

    @Slot(str)
    def _on_queue_log(self, msg):
        self._append_dash_log("MACRO", "INFO", msg)

    @Slot(str, str)
    def _on_job_log(self, jid, msg):
        if "Erro" in msg or "Falha" in msg or "Error" in msg:
            self._append_dash_log("AUDIO" if "TTS" in msg else "MACRO", "ERROR", msg)
        else:
            self._append_dash_log("AUDIO" if "TTS" in msg else "MACRO", "INFO", msg)

    def _copy_log(self):
        cb = QApplication.clipboard()
        cb.setText(self.log_html.toPlainText())
        
    # -------------------------------------------------------------------------
    # RESTO DO FORMULÁRIO (Igual à original, adaptado para tabela)
    # -------------------------------------------------------------------------

    def _start_macro(self):
        self.log_html.clear()
        self.timing.start_queue(len(self.coordinator.jobs))
        
        # Reseta tabela UI
        for jid, w in self._job_widgets.items():
            w["status_lbl"].setText("⏳ Wait")
            w["status_lbl"].setStyleSheet("color: #ccc;")
            w["aud_bar"].setValue(0)
            w["vid_bar"].setValue(0)
            w["vid_bar"].setMaximum(100)
            w["lbl_time"].setText("--:--:--")
            for c in range(self.table.columnCount()):
                item = self.table.item(w["row"], c)
                if item: item.setBackground(QColor(0, 0, 0, 0))
                
        self.coordinator.start()

    def _on_engine_changed(self):
        self.combo_model_type.clear()
        eng = self.combo_engine.currentData()
        if eng == "chatterbox":
            self.combo_model_type.addItem("Turbo (Recomendado)", "turbo")
            self.combo_model_type.addItem("Multilingual (Sotaque)", "multilingual")
            self.combo_model_type.addItem("Original (RTX 30+)", "original")
        else:
            self.combo_model_type.addItem("Kokoro Fast (24kHz)", "fast")
            self.combo_model_type.addItem("Kokoro v1.0 (Beta)", "kokoro_v1")

    @Slot()
    def _on_workflow_changed(self):
        wf = self.combo_workflow.currentData()
        self.tts_container.setVisible(wf != "video_edit")
        self.video_container.setVisible(wf in ("audio_video", "video_edit"))
        self.btn_pick_aud.setVisible(wf == "video_edit")
        self.lbl_aud_val.setVisible(wf == "video_edit")
        
    def _refresh_voices(self):
        self.combo_voice.clear()
        root_voices = Path(__file__).resolve().parent.parent / "voices"
        cloned_voices = root_voices / "cloned"
        
        potential_files = []
        if root_voices.exists():
            potential_files.extend(list(root_voices.glob("*.pt")) + list(root_voices.glob("*.wav")))
        if cloned_voices.exists():
            potential_files.extend(list(cloned_voices.glob("*.wav")) + list(cloned_voices.glob("*.mp3")))
            
        found_paths = set()
        for f in sorted(potential_files, key=lambda x: x.name.lower()):
            if str(f) in found_paths: continue
            found_paths.add(str(f))
            name = f"👤 {f.name}" if "cloned" in str(f) else f"🔊 {f.name}"
            self.combo_voice.addItem(name, str(f))

    def _pick_txt(self):
        f, _ = QFileDialog.getOpenFileName(self, "Selecionar Script", "", "Scripts (*.txt)")
        if f:
            self._current_txt = f
            self.lbl_txt_val.setText(Path(f).name)
            self.edit_proj.setText(Path(f).stem.replace(" ","_").lower())

    def _pick_aud_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Áudios")
        if d:
            self._audio_dir = d
            self.lbl_aud_val.setText(Path(d).name)
            
    def _pick_voice(self):
        f, _ = QFileDialog.getOpenFileName(self, "Selecionar Referência", "", "Audio (*.pt *.wav *.mp3)")
        if f:
            name = f"👤 {Path(f).name} (Externo)"
            idx = self.combo_voice.findData(f)
            if idx < 0:
                self.combo_voice.addItem(name, f)
                idx = self.combo_voice.count() - 1
            self.combo_voice.setCurrentIndex(idx)

    def _pick_img_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Imagens")
        if d:
            self._img_dir = d
            self.lbl_img_val.setText(Path(d).name)

    def _add_single_job(self):
        curr_wf = self.combo_workflow.currentData()
        curr_proj = self.edit_proj.text().strip()

        if curr_wf == "video_edit":
            if not curr_proj: return QMessageBox.warning(self, "Erro", "Digite o nome do Projeto.")
        else:
            if not self._current_txt: return QMessageBox.warning(self, "Erro", "Selecione script.")
        
        # Same setup as before
        audio_snapshot = {}
        if hasattr(self.main_window.audio_tab, "get_session"):
            audio_snapshot = self.main_window.audio_tab.get_session()
        if hasattr(self.main_window.tts_tab, "get_session"):
            audio_snapshot.update(self.main_window.tts_tab.get_session())
            
        video_snapshot = {}
        if hasattr(self.main_window.video_tab, "get_session"):
            video_snapshot = self.main_window.video_tab.get_session()

        import re, time
        safe_proj = re.sub(r'[^a-zA-Z0-9_\-]', '_', curr_proj or "projeto_macro")
        job_id = f"job_{int(time.time())}_{len(self.coordinator.jobs)}"
        curr_voice = self.combo_voice.currentData() or ""
        
        job = MacroJob(
            id=job_id, project_name=safe_proj, workflow=curr_wf,
            txt_path=self._current_txt, img_dir=self._img_dir,
            engine=self.combo_engine.currentData(), model_type=self.combo_model_type.currentData(),
            voice=curr_voice, lang=self.combo_lang.currentData(),
            output_root=self.main_window.audio_tab.output_root_edit.text() if hasattr(self.main_window.audio_tab, "output_root_edit") else "output",
            audio_dir=self._audio_dir if curr_wf == "video_edit" else "",
            temperature=audio_snapshot.get("temperature", 0.8),
            speed=audio_snapshot.get("speed", 1.0),
            top_p=audio_snapshot.get("top_p", 1.0),
            audio_params=audio_snapshot, video_params=video_snapshot
        )
        self.coordinator.add_job(job)
        self._add_job_row(job)

    def _bulk_import(self):
        root = QFileDialog.getExistingDirectory(self, "Selecionar Pasta Raiz")
        if not root: return
        scripts = list(Path(root).rglob("*.txt"))
        if not scripts: return QMessageBox.information(self, "Vazio", "Nenhum .txt.")
        
        # Similar à anterior ...
        import time
        audio_snapshot, video_snapshot = {}, {}
        if hasattr(self.main_window.audio_tab, "get_session"): audio_snapshot = self.main_window.audio_tab.get_session()
        if hasattr(self.main_window.tts_tab, "get_session"): audio_snapshot.update(self.main_window.tts_tab.get_session())
        if hasattr(self.main_window.video_tab, "get_session"): video_snapshot = self.main_window.video_tab.get_session()

        for i, s in enumerate(scripts):
            job = MacroJob(
                id=f"job_{int(time.time())}_{i}_{len(self.coordinator.jobs)}", project_name=s.stem.replace(" ","_").lower(),
                workflow=self.combo_workflow.currentData(), txt_path=str(s), img_dir="",
                engine=self.combo_engine.currentData(), model_type=self.combo_model_type.currentData(),
                voice=self.combo_voice.currentData() or "", lang=self.combo_lang.currentData(),
                output_root=self.main_window.audio_tab.output_root_edit.text() if hasattr(self.main_window.audio_tab, "output_root_edit") else "output",
                audio_dir="", temperature=audio_snapshot.get("temperature", 0.8),
                speed=audio_snapshot.get("speed", 1.0), top_p=audio_snapshot.get("top_p", 1.0),
                audio_params=audio_snapshot, video_params=video_snapshot
            )
            self.coordinator.add_job(job)
            self._add_job_row(job)

    def _add_job_row(self, job):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        self.table.setItem(row, 0, QTableWidgetItem(str(row+1)))
        self.table.setItem(row, 1, QTableWidgetItem(job.project_name))
        self.table.setItem(row, 2, QTableWidgetItem(job.mode_label()))
        self.table.setItem(row, 3, QTableWidgetItem(job.engine_label()))
        
        lbl_status = QLabel("⏳ Wait")
        lbl_status.setStyleSheet("color: #ccc; margin-left: 5px;")
        self.table.setCellWidget(row, 4, lbl_status)
        
        # Audio Bar
        p_aud = QProgressBar(); p_aud.setValue(0); p_aud.setFixedHeight(12)
        p_aud.setStyleSheet("QProgressBar { border-radius: 6px; text-align: center; color: transparent; background: rgba(0,188,212,0.1); border: 1px solid rgba(0,188,212,0.3); } QProgressBar::chunk { background: #00bcd4; border-radius: 5px; }")
        self.table.setCellWidget(row, 5, p_aud)
        
        # Video Bar
        p_vid = QProgressBar(); p_vid.setValue(0); p_vid.setFixedHeight(12)
        p_vid.setStyleSheet("QProgressBar { border-radius: 6px; text-align: center; color: transparent; background: rgba(255,64,129,0.1); border: 1px solid rgba(255,64,129,0.3); } QProgressBar::chunk { background: #ff4081; border-radius: 5px; }")
        self.table.setCellWidget(row, 6, p_vid)
        
        lbl_time = QLabel("--:--:--")
        lbl_time.setStyleSheet("color: #888; font-family: monospace; font-size: 11px;")
        self.table.setCellWidget(row, 7, lbl_time)
        
        action_w = QWidget()
        action_layout = QHBoxLayout(action_w)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(5)
        
        btn_play = QPushButton("▶️")
        btn_play.setFixedSize(28, 28)
        btn_play.setToolTip("Abrir Resultado / Tocar")
        btn_play.setStyleSheet("QPushButton { background: rgba(100,255,100,0.1); border: 1px solid rgba(100,255,100,0.3); border-radius: 4px; } QPushButton:hover { background: rgba(100,255,100,0.3); }")
        btn_play.clicked.connect(lambda: self._play_job(job))
        
        btn_edit = QPushButton("✏️")
        btn_edit.setFixedSize(28, 28)
        btn_edit.setToolTip("Editar Fila")
        btn_edit.setStyleSheet("QPushButton { background: rgba(100,100,255,0.1); border: 1px solid rgba(100,100,255,0.3); border-radius: 4px; } QPushButton:hover { background: rgba(100,100,255,0.3); }")
        btn_edit.clicked.connect(lambda: self._edit_job(job))

        btn_del = QPushButton("❌")
        btn_del.setFixedSize(28, 28)
        btn_del.setToolTip("Remover")
        btn_del.setStyleSheet("QPushButton { background: rgba(255,100,100,0.1); border: 1px solid rgba(255,100,100,0.3); border-radius: 4px; } QPushButton:hover { background: rgba(255,100,100,0.3); }")
        btn_del.clicked.connect(lambda: self._remove_job(job.id))
        
        action_layout.addWidget(btn_play)
        action_layout.addWidget(btn_edit)
        action_layout.addWidget(btn_del)
        self.table.setCellWidget(row, 8, action_w)
        
        self._job_widgets[job.id] = {
            "row": row, "status_lbl": lbl_status, "aud_bar": p_aud, "vid_bar": p_vid, "lbl_time": lbl_time
        }

    def _remove_job(self, jid):
        if self.coordinator.remove_job(jid):
            row = self._job_widgets[jid]["row"]
            self.table.removeRow(row)
            del self._job_widgets[jid]
            # Refresh rows
            for j in self.coordinator.jobs:
                if j.id in self._job_widgets:
                    r = self._job_widgets[j.id]["row"]
                    if r > row:
                        self._job_widgets[j.id]["row"] = r - 1
                        self.table.item(r-1, 0).setText(str(r))

    def _clear_all(self):
        if self.coordinator.clear_jobs():
            self.table.setRowCount(0)
            self._job_widgets.clear()
            self.log_html.clear()

    # Serialization (Mesmo que o original)
    def get_session(self) -> dict:
        cs = [{"text": self.combo_voice.itemText(i), "data": self.combo_voice.itemData(i)} for i in range(self.combo_voice.count())]
        return {"jobs": [j.to_dict() for j in self.coordinator.jobs], "custom_voices": cs}

    def load_session(self, data: dict):
        if "custom_voices" in data:
            self.combo_voice.clear()
            for v in data["custom_voices"]: self.combo_voice.addItem(v["text"], v["data"])
        jobs_data = data.get("jobs", [])
        if not jobs_data: return
        self._clear_all()
        for jd in jobs_data:
            job = MacroJob.from_dict(jd)
            self.coordinator.add_job(job)
            self._add_job_row(job)

    def _export_queue(self):
        f, _ = QFileDialog.getSaveFileName(self, "Exportar Fila", "", "Macro JSON (*.json)")
        if f:
            with open(f, "w", encoding="utf-8") as file: json.dump(self.get_session(), file, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "Sucesso", "Exportada.")

    def _import_queue(self):
        f, _ = QFileDialog.getOpenFileName(self, "Importar Fila", "", "Macro JSON (*.json)")
        if f:
            with open(f, "r", encoding="utf-8") as file: self.load_session(json.load(file))
            QMessageBox.information(self, "Sucesso", "Importada.")
