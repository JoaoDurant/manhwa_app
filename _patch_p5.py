"""Final targeted patch for P5 (QueueTab form) using exact match."""
import py_compile, shutil, tempfile, os, re

with open('manhwa_app/app.py', encoding='utf-8') as f:
    src = f.read()

# ── P5: Replace old form with new one ────────────────────────────────────────
qtab = src.find('class QueueTab(QWidget):')
form_s = src.find('        add_group = QGroupBox', qtab)
lv_add = src.find('        lv.addWidget(add_group)', form_s)
assert form_s > 0 and lv_add > form_s, "bounds not found"

old_form = src[form_s:lv_add]
print("Old form (100):", repr(old_form[:100]))

new_form = (
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

src = src[:form_s] + new_form + src[lv_add:]
print("P5 OK")


# ── P6: Add _pick_voice_file before _add_task ─────────────────────────────────
qtab2 = src.find('class QueueTab(QWidget):')
add_task_m = src.find('\n    def _add_task(self):', qtab2)
assert add_task_m > 0, "_add_task not found"

pick_voice_code = (
    '\n'
    '    def _pick_voice_file(self):\n'
    '        p, _ = QFileDialog.getOpenFileName(\n'
    '            self, "Selecionar Arquivo de Voz", "",\n'
    '            "Arquivos de Voz (*.pt *.wav *.mp3)"\n'
    '        )\n'
    '        if p:\n'
    '            self.q_voice_edit.setText(p)\n'
    '\n'
)

# Insert before "\n    def _add_task"
src = src[:add_task_m] + pick_voice_code + src[add_task_m + 1:]  # skip the leading \n
print("P6 OK")


# ── P7: Update _add_task QueueTask instantiation ──────────────────────────────
# Find old engine_combo reference
qtab3 = src.find('class QueueTab(QWidget):')
add2 = src.find('    def _add_task(self):', qtab3)
task_inst = src.find('        task = QueueTask(', add2)
task_end  = src.find('        )\n', task_inst) + len('        )\n')
old_inst  = src[task_inst:task_end]
print("Old inst:", repr(old_inst))

new_inst = (
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
assert old_inst in src, "P7 old_inst not in src!"
src = src.replace(old_inst, new_inst, 1)
print("P7 OK")


# ── P8: _start_queue → use QueueCoordinator ───────────────────────────────────
old_orch_inst = '        self._orchestrator = QueueOrchestrator(self._tasks, w, parent=self)\n'
new_orch_inst = '        self._orchestrator = QueueCoordinator(self._tasks, w, parent=self)\n'
if old_orch_inst in src:
    src = src.replace(old_orch_inst, new_orch_inst, 1)
    print("P8: replaced QueueOrchestrator with QueueCoordinator")
else:
    print("P8: already using QueueCoordinator")

eng_sw = '        self._orchestrator.engine_switch_needed.connect(w._handle_audio_tab_engine_switch)\n'
if eng_sw in src:
    src = src.replace(eng_sw, '', 1)
    print("P8: removed engine_switch_needed")


# ── Verify & write ─────────────────────────────────────────────────────────────
tmp = tempfile.mktemp(suffix='.py')
with open(tmp, 'w', encoding='utf-8') as f:
    f.write(src)

try:
    py_compile.compile(tmp, doraise=True)
    print("\nSYNTAX OK")
    shutil.copy(tmp, 'manhwa_app/app.py')
    print(f"Saved! ({len(src)} bytes)")
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
