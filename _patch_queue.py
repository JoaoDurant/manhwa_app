"""
Queue redesign patch - LF line endings, no \r.
All string literals use plain \n.
"""
import py_compile, shutil, tempfile, os, re

with open('manhwa_app/app.py', encoding='utf-8') as f:
    src = f.read()

assert '\r' not in src, "File has CR bytes!"
print(f"File: {src.count(chr(10))} lines, LF only")


# ═══════════════════════════════════════════════════════════════════════════
# P1 - QueueTask: add lang_override parameter and field
# ═══════════════════════════════════════════════════════════════════════════
P1_OLD = (
    '                 voice_override: str = None, img_path: str = None):\n'
    '        self.project_name        = project_name\n'
    '        self.txt_path            = txt_path\n'
    '        self.mode                = mode\n'
    '        self.engine_override     = engine_override\n'
    '        self.model_type_override = model_type_override\n'
    '        self.voice_override      = voice_override\n'
    '        self.img_path            = img_path\n'
)
P1_NEW = (
    '                 voice_override: str = None, lang_override: str = None,\n'
    '                 img_path: str = None):\n'
    '        self.project_name        = project_name\n'
    '        self.txt_path            = txt_path\n'
    '        self.mode                = mode\n'
    '        self.engine_override     = engine_override\n'
    '        self.model_type_override = model_type_override\n'
    '        self.voice_override      = voice_override\n'
    '        self.lang_override       = lang_override\n'
    '        self.img_path            = img_path\n'
)
assert P1_OLD in src, "P1 not found"
src = src.replace(P1_OLD, P1_NEW, 1)
print("P1 OK")


# ═══════════════════════════════════════════════════════════════════════════
# P2 - Replace QueueOrchestrator class with QueueCoordinator
# ═══════════════════════════════════════════════════════════════════════════
# Find exact bounds
idx_orch = src.find('\nclass QueueOrchestrator(QThread):')
# The class ends just before the DashboardTab separator comment
idx_dash_sep = src.find('\n# ---------------------------------------------------------------------------\n# Dashboard Tab', idx_orch)
assert idx_orch > 0 and idx_dash_sep > idx_orch, "Class bounds not found"
print(f"P2: replacing chars {idx_orch}..{idx_dash_sep}")

NEW_COORD = '''
class QueueCoordinator(QObject):
    """
    Coordenador de fila event-driven.
    Delega a geração para o pipeline da aba Áudio (não tem pipeline próprio).
    """
    task_started  = Signal(int)           # task_idx
    task_progress = Signal(int, int, int) # task_idx, current, total
    task_log      = Signal(int, str)      # task_idx, message
    task_finished = Signal(int, bool)     # task_idx, success
    queue_log     = Signal(str)
    all_done      = Signal()

    def __init__(self, tasks: list, main_window, parent=None):
        super().__init__(parent)
        self.tasks       = list(tasks)
        self.main_window = main_window
        self._idx        = 0
        self._cancelled  = False
        self._audio_paths: list = []

    def cancel(self):
        self._cancelled = True
        at = self.main_window.audio_tab
        if at._pipeline:
            at._pipeline.cancel()

    # ── Ponto de entrada ─────────────────────────────────────────────────
    def start(self):
        if not self.tasks:
            self.all_done.emit()
            return
        self._idx = 0
        self._cancelled = False
        self._connect_audio()
        self._run_current()

    def _connect_audio(self):
        at = self.main_window.audio_tab
        at.pipeline_progress.connect(self._on_progress)
        at.pipeline_log.connect(self._on_log)
        at.pipeline_finished.connect(self._on_audio_done)
        at.audio_generated.connect(self._on_generated)

    def _disconnect_audio(self):
        at = self.main_window.audio_tab
        for sig, slot in [
            (at.pipeline_progress, self._on_progress),
            (at.pipeline_log,      self._on_log),
            (at.pipeline_finished, self._on_audio_done),
            (at.audio_generated,   self._on_generated),
        ]:
            try: sig.disconnect(slot)
            except: pass

    # ── Máquina de estados ───────────────────────────────────────────────
    def _run_current(self):
        if self._cancelled or self._idx >= len(self.tasks):
            self._finish()
            return
        task = self.tasks[self._idx]
        task.status = "running"
        self.task_started.emit(self._idx)
        self.queue_log.emit(
            f"\\n\\U0001f680 [{self._idx+1}/{len(self.tasks)}] "
            f"{task.project_name} ({task.mode_label()})"
        )
        self.main_window.audio_tab.configure_and_run_queued(task)

    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        self.task_progress.emit(self._idx, current, total)

    @Slot(str)
    def _on_log(self, msg: str):
        self.task_log.emit(self._idx, msg)

    @Slot(list)
    def _on_generated(self, paths: list):
        self._audio_paths = paths

    @Slot(bool, str)
    def _on_audio_done(self, success: bool, msg: str):
        task = self.tasks[self._idx]
        if not success:
            task.status = "error"
            self.queue_log.emit(f"\\u274c Erro: {msg}")
            self.task_finished.emit(self._idx, False)
            self._idx += 1
            self._run_current()
            return
        if task.mode in ("audio+video", "audio+video+alt"):
            self._start_video(task)
        else:
            task.status = "done"
            self.task_finished.emit(self._idx, True)
            self._idx += 1
            self._run_current()

    def _start_video(self, task: "QueueTask"):
        from manhwa_app.video_pipeline import VideoPipeline
        w        = self.main_window
        out_root = w.audio_tab.output_root_edit.text().strip() or "output"
        if task.mode == "audio+video+alt" and task.img_path:
            images = sorted(
                [str(p) for p in Path(task.img_path).glob("*")
                 if p.suffix.lower() in (".jpg", ".png", ".webp", ".jpeg")],
                key=natural_sort_key
            )
        else:
            images = w.images_tab.get_images()
        audios_dir = Path(out_root) / task.project_name / "audios"
        audios = sorted(
            [str(p) for p in audios_dir.glob("audio_*.wav")],
            key=lambda x: int(Path(x).stem.split("_")[1])
                          if Path(x).stem.split("_")[1].isdigit() else 0
        )
        if not (audios and images):
            self.task_log.emit(self._idx, "\\u26a0 Sem áudios/imagens para vídeo.")
            task.status = "done"
            self.task_finished.emit(self._idx, True)
            self._idx += 1
            self._run_current()
            return
        vcfg = w.video_tab.get_session()
        pairs = list(zip(audios, images[:len(audios)]))
        out_vid = str(
            Path(out_root) / task.project_name / f"{task.project_name}_final.mp4"
        )
        self._v_pipe = VideoPipeline(
            pairs=pairs, output_path=out_vid,
            effect_mode=vcfg.get("effect", "auto"),
            transition_mode=vcfg.get("transition", "fade"),
            transition_time=vcfg.get("transition_time", 0.5),
        )
        self._v_pipe.log_message.connect(
            lambda m, i=self._idx: self.task_log.emit(i, m)
        )
        self._v_thread = QThread(self)
        self._v_pipe.moveToThread(self._v_thread)
        self._v_thread.started.connect(self._v_pipe.run)
        self._v_pipe.finished.connect(self._on_video_done)
        self._v_thread.start()

    @Slot(bool, str)
    def _on_video_done(self, success: bool, msg: str):
        self._v_thread.quit()
        task = self.tasks[self._idx]
        task.status = "done" if success else "error"
        self.task_finished.emit(self._idx, success)
        self._idx += 1
        self._run_current()

    def _finish(self):
        self._disconnect_audio()
        self.queue_log.emit("\\n\\U0001f3af Fila concluída!")
        self.all_done.emit()


# Alias para compatibilidade com código existente
QueueOrchestrator = QueueCoordinator
'''

src = src[:idx_orch] + NEW_COORD + src[idx_dash_sep:]
print("P2 OK")


# ═══════════════════════════════════════════════════════════════════════════
# P3 - AudioTab: add configure_and_run_queued() before get_generated_paths
# ═══════════════════════════════════════════════════════════════════════════
ANCHOR_P3 = '    def get_generated_paths(self) -> List[str]:'
assert ANCHOR_P3 in src, "P3 anchor not found"

NEW_CFG = (
    '    def configure_and_run_queued(self, task: "QueueTask"):\n'
    '        """Configura a aba Áudio para uma tarefa da fila e inicia geração."""\n'
    '        self.project_edit.setText(task.project_name)\n'
    '        lang  = task.lang_override  or self.preset_lang_combo.currentText()\n'
    '        voice = task.voice_override or ""\n'
    '        self._files = [{"path": task.txt_path, "voice": voice, "lang": lang}]\n'
    '        self._refresh_list()\n'
    '        if task.lang_override:\n'
    '            idx = self.preset_lang_combo.findText(task.lang_override)\n'
    '            if idx >= 0: self.preset_lang_combo.setCurrentIndex(idx)\n'
    '        if task.voice_override:\n'
    '            idx = self.preset_voice_combo.findData(task.voice_override)\n'
    '            if idx >= 0: self.preset_voice_combo.setCurrentIndex(idx)\n'
    '        self._queued_overrides = task  # lido em _start_pipeline\n'
    '        self._start_normal()\n'
    '\n'
    '    def get_generated_paths(self) -> List[str]:\n'
)
src = src.replace(ANCHOR_P3, NEW_CFG, 1)
print("P3 OK")


# ═══════════════════════════════════════════════════════════════════════════
# P4 - _start_pipeline: consume _queued_overrides
# ═══════════════════════════════════════════════════════════════════════════
P4_OLD = (
    '        cfg = self._get_tts_config()\n'
    '        \n'
    '        voice_val = self.preset_voice_combo.currentData()\n'
)
P4_NEW = (
    '        cfg = self._get_tts_config()\n'
    '\n'
    '        # Overrides da fila (engine / submodelo)\n'
    '        if hasattr(self, "_queued_overrides") and self._queued_overrides:\n'
    '            _ot = self._queued_overrides\n'
    '            if _ot.engine_override:     cfg["tts_engine"] = _ot.engine_override\n'
    '            if _ot.model_type_override: cfg["model_type"] = _ot.model_type_override\n'
    '            self._queued_overrides = None\n'
    '\n'
    '        voice_val = self.preset_voice_combo.currentData()\n'
)
assert P4_OLD in src, "P4 anchor not found"
src = src.replace(P4_OLD, P4_NEW, 1)
print("P4 OK")


# ═══════════════════════════════════════════════════════════════════════════
# P5 - QueueTab form: replace content of add_group
# ═══════════════════════════════════════════════════════════════════════════
# Bounds: from "add_group = QGroupBox" up to (not including) "lv.addWidget(add_group)"
qtab_cls = src.find('class QueueTab(QWidget):')
form_start = src.find('        add_group = QGroupBox("\u2795 Adicionar Tarefa")', qtab_cls)
form_end   = src.find('        lv.addWidget(add_group)', form_start)
assert form_start > 0 and form_end > form_start, "P5 bounds not found"
print(f"P5 form bounds: {form_start}..{form_end}")

NEW_FORM = (
    '        add_group = QGroupBox("\u2795 Adicionar Tarefa")\n'
    '        fl = QGridLayout(add_group)\n'
    '        fl.setSpacing(8)\n'
    '\n'
    '        row = 0\n'
    '        fl.addWidget(QLabel("Nome do Projeto:"), row, 0)\n'
    '        self.proj_edit = QLineEdit("projeto_fila")\n'
    '        fl.addWidget(self.proj_edit, row, 1, 1, 2); row += 1\n'
    '\n'
    '        fl.addWidget(QLabel("Arquivo .txt:"), row, 0)\n'
    '        self.btn_txt = QPushButton("\U0001f4c4 Selecionar\u2026")\n'
    '        self.btn_txt.clicked.connect(self._pick_txt)\n'
    '        fl.addWidget(self.btn_txt, row, 1, 1, 2); row += 1\n'
    '        self.lbl_txt = QLabel("(nenhum)")\n'
    '        self.lbl_txt.setStyleSheet("color:#888; font-size:10px;")\n'
    '        fl.addWidget(self.lbl_txt, row, 0, 1, 3); row += 1\n'
    '\n'
    '        fl.addWidget(QLabel("Modo:"), row, 0)\n'
    '        self.mode_combo = QComboBox()\n'
    '        for k, v in QueueTask.MODES.items():\n'
    '            self.mode_combo.addItem(v, k)\n'
    '        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)\n'
    '        fl.addWidget(self.mode_combo, row, 1, 1, 2); row += 1\n'
    '\n'
    '        # Engine + Submodelo\n'
    '        fl.addWidget(QLabel("Engine:"), row, 0)\n'
    '        self.q_engine_combo = QComboBox()\n'
    '        self.q_engine_combo.addItem("\u2014 Usar Atual \u2014", None)\n'
    '        for eng in ["chatterbox", "kokoro"]:\n'
    '            self.q_engine_combo.addItem(eng, eng)\n'
    '        fl.addWidget(self.q_engine_combo, row, 1)\n'
    '        self.q_model_combo = QComboBox()\n'
    '        self.q_model_combo.addItem("\u2014 Submodelo Atual \u2014", None)\n'
    '        self.q_model_combo.addItem("turbo", "turbo")\n'
    '        self.q_model_combo.addItem("fast", "fast")\n'
    '        fl.addWidget(self.q_model_combo, row, 2); row += 1\n'
    '\n'
    '        # Voz\n'
    '        fl.addWidget(QLabel("Voz (ID / Arquivo):"), row, 0)\n'
    '        self.q_voice_edit = QLineEdit()\n'
    '        self.q_voice_edit.setPlaceholderText("ex: pf_dora  (vazio = usar aba \u00c1udio)")\n'
    '        fl.addWidget(self.q_voice_edit, row, 1)\n'
    '        _btn_vb = QPushButton("\U0001f4c1")\n'
    '        _btn_vb.setFixedWidth(30)\n'
    '        _btn_vb.setToolTip("Escolher arquivo de voz (.pt / .wav)")\n'
    '        _btn_vb.clicked.connect(self._pick_voice_file)\n'
    '        fl.addWidget(_btn_vb, row, 2); row += 1\n'
    '\n'
    '        # Idioma\n'
    '        fl.addWidget(QLabel("Idioma:"), row, 0)\n'
    '        self.q_lang_combo = QComboBox()\n'
    '        self.q_lang_combo.addItem("\u2014 Usar Atual \u2014", None)\n'
    '        for lg in ["pt", "en", "es", "fr", "ja", "ko", "zh"]:\n'
    '            self.q_lang_combo.addItem(lg, lg)\n'
    '        fl.addWidget(self.q_lang_combo, row, 1, 1, 2); row += 1\n'
    '\n'
    '        # Pasta imagens alternativas\n'
    '        self.lbl_img = QLabel("Pasta de Imagens:")\n'
    '        fl.addWidget(self.lbl_img, row, 0)\n'
    '        self.img_row = QWidget()\n'
    '        ir = QHBoxLayout(self.img_row)\n'
    '        ir.setContentsMargins(0, 0, 0, 0)\n'
    '        self.btn_imgs = QPushButton("\U0001f5bc Selecionar\u2026")\n'
    '        self.btn_imgs.clicked.connect(self._pick_imgs)\n'
    '        ir.addWidget(self.btn_imgs)\n'
    '        fl.addWidget(self.img_row, row, 1, 1, 2); row += 1\n'
    '        self.lbl_imgs = QLabel("(usar aba Imagens)")\n'
    '        self.lbl_imgs.setStyleSheet("color:#888; font-size:10px;")\n'
    '        fl.addWidget(self.lbl_imgs, row, 0, 1, 3); row += 1\n'
    '\n'
    '        btn_add = QPushButton("\u2795 Adicionar \u00e0 Fila")\n'
    '        btn_add.setObjectName("primary")\n'
    '        btn_add.setMinimumHeight(36)\n'
    '        btn_add.clicked.connect(self._add_task)\n'
    '        fl.addWidget(btn_add, row, 0, 1, 3)\n'
    '        '
)
src = src[:form_start] + NEW_FORM + src[form_end:]
print("P5 OK")


# ═══════════════════════════════════════════════════════════════════════════
# P6 - Add _pick_voice_file method to QueueTab (before _add_task)
# ═══════════════════════════════════════════════════════════════════════════
qtab2 = src.find('class QueueTab(QWidget):')
add_task_pos = src.find('\n    def _add_task(self):', qtab2)
assert add_task_pos > 0, "_add_task not found in QueueTab"

PICK_VOICE_M = (
    '\n'
    '    def _pick_voice_file(self):\n'
    '        p, _ = QFileDialog.getOpenFileName(\n'
    '            self, "Selecionar Arquivo de Voz", "",\n'
    '            "Arquivos de Voz (*.pt *.wav *.mp3)"\n'
    '        )\n'
    '        if p:\n'
    '            self.q_voice_edit.setText(p)\n'
    '\n'
    '    def _add_task(self):\n'
)
src = src[:add_task_pos] + PICK_VOICE_M + src[add_task_pos + len('\n    def _add_task(self):'):]
print("P6 OK")


# ═══════════════════════════════════════════════════════════════════════════
# P7 - _add_task: update QueueTask instantiation
# ═══════════════════════════════════════════════════════════════════════════
P7_OLD = (
    '        task = QueueTask(\n'
    '            project_name=self.proj_edit.text().strip() or "projeto",\n'
    '            txt_path=self._current_txt,\n'
    '            mode=mode,\n'
    '            engine_override=self.engine_combo.currentData(),\n'
    '            img_path=self._alt_img_path if mode == "audio+video+alt" else None,\n'
    '        )\n'
)
P7_NEW = (
    '        task = QueueTask(\n'
    '            project_name=self.proj_edit.text().strip() or "projeto",\n'
    '            txt_path=self._current_txt,\n'
    '            mode=mode,\n'
    '            engine_override=self.q_engine_combo.currentData(),\n'
    '            model_type_override=self.q_model_combo.currentData(),\n'
    '            voice_override=self.q_voice_edit.text().strip() or None,\n'
    '            lang_override=self.q_lang_combo.currentData(),\n'
    '            img_path=self._alt_img_path if mode == "audio+video+alt" else None,\n'
    '        )\n'
)
assert P7_OLD in src, "P7 not found"
src = src.replace(P7_OLD, P7_NEW, 1)
print("P7 OK")


# ═══════════════════════════════════════════════════════════════════════════
# P8 - _start_queue: use QueueCoordinator, remove engine_switch_needed
# ═══════════════════════════════════════════════════════════════════════════
P8_OLD = '        self._orchestrator = QueueOrchestrator(self._tasks, w, parent=self)\n'
P8_NEW = '        self._orchestrator = QueueCoordinator(self._tasks, w, parent=self)\n'
assert P8_OLD in src, "P8 not found"
src = src.replace(P8_OLD, P8_NEW, 1)

eng_sw = '        self._orchestrator.engine_switch_needed.connect(w._handle_audio_tab_engine_switch)\n'
if eng_sw in src:
    src = src.replace(eng_sw, '', 1)
    print("P8 OK - engine_switch_needed removed")
else:
    print("P8 OK - engine_switch_needed already gone")


# ═══════════════════════════════════════════════════════════════════════════
# Verify & write
# ═══════════════════════════════════════════════════════════════════════════
tmp = tempfile.mktemp(suffix='.py')
with open(tmp, 'w', encoding='utf-8') as f:
    f.write(src)

try:
    py_compile.compile(tmp, doraise=True)
    print("\nSYNTAX OK")
    shutil.copy(tmp, 'manhwa_app/app.py')
    print(f"manhwa_app/app.py saved! ({len(src)} bytes)")
except py_compile.PyCompileError as e:
    err_str = str(e)
    print(f"\nSYNTAX ERROR: {err_str}")
    m = re.search(r'line (\d+)', err_str)
    if m:
        ln = int(m.group(1))
        flines = src.splitlines()
        for i, l in enumerate(flines[max(0,ln-6):ln+5], max(1,ln-5)):
            print(f"{'>>>' if i==ln else '   '} {i:4d}: {l}")
    raise SystemExit(1)
finally:
    try: os.unlink(tmp)
    except: pass
