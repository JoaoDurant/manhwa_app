"""
Queue redesign execution script. 
1. Updates QueueTab UI and methods.
2. Updates AudioTab for queue integration.
"""
import py_compile, shutil, tempfile, os

with open('manhwa_app/app.py', encoding='utf-8') as f:
    src = f.read()

# ─────────────────────────────────────────────────────────────────────────────
# P5: QueueTab rebuild (UI + Methods)
# ─────────────────────────────────────────────────────────────────────────────
qtab_start = src.find('class QueueTab(QWidget):')
qtab_next_class = src.find('\nclass MainWindow', qtab_start)
if qtab_next_class == -1: qtab_next_class = len(src)

NEW_QTAB_CODE = '''class QueueTab(QWidget):
    """Aba de controle da fila de geração."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: list = []       # list of QueueTask
        self._orchestrator = None
        self._current_txt: str = ""
        self._alt_img_path: str = ""
        self._setup_ui()

    def _setup_ui(self):
        lv = QVBoxLayout(self)
        lv.setContentsMargins(15, 15, 15, 15)
        lv.setSpacing(12)

        # ── Painel de Adição (GroupBox) ───────────────────────────────────
        add_group = QGroupBox("➕ Adicionar Tarefa")
        fl = QGridLayout(add_group)
        fl.setSpacing(10)

        row = 0
        fl.addWidget(QLabel("Nome do Projeto:"), row, 0)
        self.proj_edit = QLineEdit("projeto_fila")
        fl.addWidget(self.proj_edit, row, 1, 1, 2); row += 1

        fl.addWidget(QLabel("Arquivo .txt:"), row, 0)
        self.btn_txt = QPushButton("📄 Selecionar…")
        self.btn_txt.clicked.connect(self._pick_txt)
        fl.addWidget(self.btn_txt, row, 1, 1, 2); row += 1
        self.lbl_txt = QLabel("(nenhum)")
        self.lbl_txt.setStyleSheet("color:#888; font-size:10px;")
        fl.addWidget(self.lbl_txt, row, 0, 1, 3); row += 1

        fl.addWidget(QLabel("Modo:"), row, 0)
        self.mode_combo = QComboBox()
        for k, v in QueueTask.MODES.items():
            self.mode_combo.addItem(v, k)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        fl.addWidget(self.mode_combo, row, 1, 1, 2); row += 1

        # ── Engine + Submodelo ───────────────────────────────────────────
        fl.addWidget(QLabel("Engine Override:"), row, 0)
        self.q_engine_combo = QComboBox()
        self.q_engine_combo.addItem("— Usar Atual —", None)
        for eng in ["chatterbox", "kokoro"]:
            self.q_engine_combo.addItem(eng, eng)
        fl.addWidget(self.q_engine_combo, row, 1)

        self.q_model_combo = QComboBox()
        self.q_model_combo.addItem("— Submodelo Atual —", None)
        self.q_model_combo.addItem("turbo", "turbo")
        self.q_model_combo.addItem("fast", "fast")
        fl.addWidget(self.q_model_combo, row, 2); row += 1

        # ── Voz ─────────────────────────────────────────────────────────
        fl.addWidget(QLabel("Voz (ID ou Caminho):"), row, 0)
        self.q_voice_edit = QLineEdit()
        self.q_voice_edit.setPlaceholderText("ex: pf_dora / voice.pt")
        fl.addWidget(self.q_voice_edit, row, 1)
        btn_v_browse = QPushButton("📁")
        btn_v_browse.setFixedWidth(35)
        btn_v_browse.clicked.connect(self._pick_voice_file)
        fl.addWidget(btn_v_browse, row, 2); row += 1

        # ── Idioma ───────────────────────────────────────────────────────
        fl.addWidget(QLabel("Idioma:"), row, 0)
        self.q_lang_combo = QComboBox()
        self.q_lang_combo.addItem("— Usar Atual —", None)
        for lg in ["pt", "en", "es", "fr", "ja", "ko", "zh"]:
            self.q_lang_combo.addItem(lg, lg)
        fl.addWidget(self.q_lang_combo, row, 1, 1, 2); row += 1

        # ── Imagens ─────────────────────────────────────────────────────
        self.lbl_img = QLabel("Pasta de Imagens:")
        fl.addWidget(self.lbl_img, row, 0)
        self.img_row = QWidget()
        ir = QHBoxLayout(self.img_row)
        ir.setContentsMargins(0, 0, 0, 0)
        self.btn_imgs = QPushButton("🖼 Selecionar…")
        self.btn_imgs.clicked.connect(self._pick_imgs)
        self.lbl_imgs = QLabel("(aba Imagens)")
        self.lbl_imgs.setStyleSheet("color:#888; font-size:10px;")
        ir.addWidget(self.btn_imgs)
        fl.addWidget(self.img_row, row, 1, 1, 2); row += 1
        fl.addWidget(self.lbl_imgs, row, 0, 1, 3); row += 1

        btn_add = QPushButton("➕ Adicionar à Fila")
        btn_add.setObjectName("primary")
        btn_add.setMinimumHeight(40)
        btn_add.clicked.connect(self._add_task)
        fl.addWidget(btn_add, row, 0, 1, 3); row += 1

        lv.addWidget(add_group)

        # ── Painel da Fila (Scroll) ───────────────────────────────────────
        lv.addWidget(QLabel("📋 Tarefas Pendentes:"))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.task_list_container = QWidget()
        self._list_layout = QVBoxLayout(self.task_list_container)
        self._list_layout.setContentsMargins(5, 5, 5, 5)
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch()
        self.scroll.setWidget(self.task_list_container)
        lv.addWidget(self.scroll)

        # ── Controles ────────────────────────────────────────────────────
        ctrl_group = QWidget()
        cl = QHBoxLayout(ctrl_group)
        cl.setContentsMargins(0, 5, 0, 0)
        
        self.btn_start = QPushButton("🚀 Iniciar Fila")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumHeight(36)
        self.btn_start.clicked.connect(self._start_queue)
        
        self.btn_stop = QPushButton("🛑 Parar")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_queue)

        cl.addWidget(self.btn_start)
        cl.addWidget(self.btn_stop)
        lv.addWidget(ctrl_group)

    def _pick_txt(self):
        f, _ = QFileDialog.getOpenFileName(self, "Arquivo de Texto", "", "Text Files (*.txt)")
        if f:
            self._current_txt = f
            self.lbl_txt.setText(Path(f).name)
            p_name = Path(f).stem
            p_name = p_name.replace(" ", "_").lower()
            self.proj_edit.setText(p_name)

    def _pick_imgs(self):
        f = QFileDialog.getExistingDirectory(self, "Pasta de Imagens")
        if f:
            self._alt_img_path = f
            self.lbl_imgs.setText(Path(f).name)

    def _pick_voice_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Arquivo de Voz", "", "Voz (*.pt *.wav *.mp3 *.pth)")
        if f:
            self.q_voice_edit.setText(f)

    def _on_mode_changed(self):
        m = self.mode_combo.currentData()
        is_alt = (m == "audio+video+alt")
        self.lbl_img.setVisible(is_alt)
        self.img_row.setVisible(is_alt)
        self.lbl_imgs.setVisible(is_alt)

    def _add_task(self):
        if not self._current_txt:
            QMessageBox.warning(self, "Erro", "Selecione um arquivo .txt")
            return
        
        mode = self.mode_combo.currentData()
        voice_val = self.q_voice_edit.text().strip() or None
        
        task = QueueTask(
            project_name=self.proj_edit.text().strip() or "projeto",
            txt_path=self._current_txt,
            mode=mode,
            engine_override=self.q_engine_combo.currentData(),
            model_type_override=self.q_model_combo.currentData(),
            voice_override=voice_val,
            lang_override=self.q_lang_combo.currentData(),
            img_path=self._alt_img_path if mode == "audio+video+alt" else None
        )
        self._tasks.append(task)
        self._refresh_list()
        
    def _refresh_list(self):
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        
        for i, task in enumerate(self._tasks):
            w = QWidget()
            w.setStyleSheet("background: rgba(255,255,255,0.05); border-radius: 4px;")
            hl = QHBoxLayout(w)
            icon = "⏳ " if task.status == "pending" else "⚡ " if task.status == "running" else "✅ " if task.status == "done" else "❌ "
            hl.addWidget(QLabel(f"{icon} <b>{task.project_name}</b> ({task.mode_label()})"))
            hl.addStretch()
            btn_del = QPushButton("🗑")
            btn_del.setFixedWidth(30)
            btn_del.clicked.connect(lambda _, idx=i: self._on_remove(idx))
            hl.addWidget(btn_del)
            self._list_layout.insertWidget(self._list_layout.count()-1, w)

    def _on_remove(self, idx):
        if 0 <= idx < len(self._tasks):
            self._tasks.pop(idx)
            self._refresh_list()

    def _start_queue(self):
        if not self._tasks:
            QMessageBox.information(self, "Vazio", "Adicione tarefas à fila primeiro.")
            return
        w = self.window()
        self._orchestrator = QueueCoordinator(self._tasks, w, parent=self)
        if hasattr(w, "dashboard_tab"):
            db = w.dashboard_tab
            db.rebuild_task_rows(self._tasks)
            self._orchestrator.task_started.connect(db.on_task_started)
            self._orchestrator.task_progress.connect(db.on_task_progress)
            self._orchestrator.task_log.connect(db.on_task_log)
            self._orchestrator.task_finished.connect(db.on_task_finished)
            self._orchestrator.queue_log.connect(db.on_queue_log)
        self._orchestrator.task_started.connect(self._on_task_started)
        self._orchestrator.task_finished.connect(self._on_task_finished)
        self._orchestrator.all_done.connect(self._on_all_done)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        if hasattr(w, "dashboard_tab"): w.tabs.setCurrentWidget(w.dashboard_tab)
        self._orchestrator.start()

    def _stop_queue(self):
        if self._orchestrator:
            self._orchestrator.cancel()
            self.btn_stop.setEnabled(False)

    def _on_task_started(self, idx): self._refresh_list()
    def _on_task_finished(self, idx, success): self._refresh_list()
    def _on_all_done(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.information(self, "Fila", "Todas as tarefas foram concluídas!")

'''

src = src[:qtab_start] + NEW_QTAB_CODE + src[qtab_next_class:]

# ─────────────────────────────────────────────────────────────────────────────
# P6: AudioTab updates
# ─────────────────────────────────────────────────────────────────────────────
# 1. Add configure_and_run_queued before get_generated_paths
gp_anchor = '    def get_generated_paths(self) -> List[str]:'
NEW_CONF_METHOD = '''    def configure_and_run_queued(self, task: "QueueTask"):
        """Configura a aba Áudio para uma tarefa da fila e inicia geração."""
        self.project_edit.setText(task.project_name)
        lang  = task.lang_override  or self.preset_lang_combo.currentText()
        voice = task.voice_override or ""
        self._files = [{"path": task.txt_path, "voice": voice, "lang": lang}]
        self._refresh_list()
        if task.lang_override:
            idx = self.preset_lang_combo.findText(task.lang_override)
            if idx >= 0: self.preset_lang_combo.setCurrentIndex(idx)
        if task.voice_override:
            idx = self.preset_voice_combo.findData(task.voice_override)
            if idx >= 0: self.preset_voice_combo.setCurrentIndex(idx)
        self._queued_overrides = task
        self._start_normal()

'''
src = src.replace(gp_anchor, NEW_CONF_METHOD + gp_anchor)

# 2. Update _start_pipeline to consume _queued_overrides
p4_old = '        cfg = self._get_tts_config()'
p4_new = '''        cfg = self._get_tts_config()

        # Overrides da fila (engine / submodelo)
        if hasattr(self, "_queued_overrides") and self._queued_overrides:
            _ot = self._queued_overrides
            if _ot.engine_override:      cfg["tts_engine"] = _ot.engine_override
            if _ot.model_type_override:  cfg["model_type"] = _ot.model_type_override
            self._queued_overrides = None
'''
src = src.replace(p4_old, p4_new, 1)

with open('manhwa_app/app.py', 'w', encoding='utf-8') as f:
    f.write(src)

print('Queues and AudioTab updated.')
'''
