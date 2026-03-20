import sys

NEW_AUDIO_TAB = """class AudioTab(QWidget):
    audio_generated = Signal(list)
    run_all_audio_done = Signal(int, str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files = []
        self._worker_thread = None
        self._pipeline = None
        self._generated = []
        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        self._audio_out.setVolume(1.0)
        self._run_all_mode = False
        self._run_all_index = 0
        self._current_run_all_project = ""
        self._playing_row = None
        self._setup_ui()
        self._player.playbackStateChanged.connect(self._on_playback_state)

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        left = QWidget()
        left.setMaximumWidth(420)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)
        proj_group = QGroupBox("Projeto & Saída")
        pg = QGridLayout(proj_group)
        pg.addWidget(QLabel("Nome do Projeto:"), 0, 0)
        self.project_edit = QLineEdit("meu_projeto")
        pg.addWidget(self.project_edit, 0, 1)
        pg.addWidget(QLabel("Pasta Raiz:"), 1, 0)
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        self.output_root_edit = QLineEdit("output")
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(30)
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(self.output_root_edit)
        out_row.addWidget(btn_out)
        pg.addLayout(out_row, 1, 1)
        lv.addWidget(proj_group)
        files_group = QGroupBox("Arquivos de Texto (.txt)")
        fv = QVBoxLayout(files_group)
        tb = QHBoxLayout()
        btn_add = QPushButton("＋ Add .txt")
        btn_add.clicked.connect(self._add_files)
        btn_folder = QPushButton("📁 Add Pasta")
        btn_folder.clicked.connect(self._add_folder)
        btn_clear = QPushButton("✖ Limpar")
        btn_clear.clicked.connect(self._clear_files)
        tb.addWidget(btn_add)
        tb.addWidget(btn_folder)
        tb.addWidget(btn_clear)
        fv.addLayout(tb)
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        fv.addWidget(self.files_list)
        vox_row = QHBoxLayout()
        vox_row.addWidget(QLabel("Voz:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Sem voz (padrão)", None)
        self._populate_voices()
        vox_row.addWidget(self.voice_combo, 1)
        vox_row.addWidget(QLabel("Idioma:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["en", "pt", "es", "fr", "ja", "ko", "zh"])
        vox_row.addWidget(self.lang_combo)
        btn_apply = QPushButton("Aplicar ao Selecionado")
        btn_apply.clicked.connect(self._apply_voice)
        vox_row.addWidget(btn_apply)
        fv.addLayout(vox_row)
        lv.addWidget(files_group)
        actions = QHBoxLayout()
        self.btn_generate = QPushButton("⚡ Gerar Áudios")
        self.btn_generate.setObjectName("primary")
        self.btn_generate.setMinimumHeight(42)
        self.btn_generate.clicked.connect(self._start_normal)
        actions.addWidget(self.btn_generate)
        self.btn_run_all = QPushButton("🚀 Run All (Áudio+Vídeo)")
        self.btn_run_all.setObjectName("primary")
        self.btn_run_all.setStyleSheet("background-color: #059669; border-color: #047857;")
        self.btn_run_all.setMinimumHeight(42)
        self.btn_run_all.clicked.connect(self._start_run_all)
        actions.addWidget(self.btn_run_all)
        self.btn_cancel = QPushButton("✖ Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setMinimumHeight(42)
        self.btn_cancel.clicked.connect(self._cancel_generation)
        actions.addWidget(self.btn_cancel)
        lv.addLayout(actions)
        root.addWidget(left)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)
        prog_group = QGroupBox("Progresso")
        pgv = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        pgv.addWidget(self.progress_bar)
        rv.addWidget(prog_group)
        log_group = QGroupBox("Log")
        logv = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("background:#171717; color:#d0d0d0; border-radius: 4px;")
        logv.addWidget(self.log_text)
        rv.addWidget(log_group)
        out_group = QGroupBox("Áudios Gerados")
        ov = QVBoxLayout(out_group)
        self.audio_list = QListWidget()
        self.audio_list.setMaximumHeight(150)
        ov.addWidget(self.audio_list)
        player_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Reproduzir Atual")
        self.btn_play_chain = QPushButton("⏭ Reproduzir Todos")
        self.btn_stop = QPushButton("⏹ Parar")
        self.btn_regen = QPushButton("↻ Regravar Atual")
        self.btn_open_folder = QPushButton("📁 Abrir Saída")
        self.btn_play.clicked.connect(self._play_selected)
        self.btn_play_chain.clicked.connect(self._play_chain)
        self.btn_stop.clicked.connect(self._stop_audio)
        self.btn_regen.clicked.connect(self._regen_selected)
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        player_row.addWidget(self.btn_play)
        player_row.addWidget(self.btn_play_chain)
        player_row.addWidget(self.btn_stop)
        player_row.addWidget(self.btn_regen)
        player_row.addStretch()
        player_row.addWidget(self.btn_open_folder)
        ov.addLayout(player_row)
        rv.addWidget(out_group)
        root.addWidget(right)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Selecionar diretório de saída")
        if d: self.output_root_edit.setText(d)

    def _populate_voices(self):
        base_dir = Path("voices").resolve()
        if not base_dir.exists(): return
        cloned, standard = [], []
        for p in base_dir.rglob("*.wav"):
            rel = p.relative_to(base_dir)
            if rel.parts[0] == "cloned":
                cloned.append(p)
            elif rel.parts[0] == "standard":
                standard.append(p)
        def format_name(p):
            name = p.stem.replace("_", " ").title()
            lang = p.parent.name.upper()
            return f"[{lang}] {name}"
        for path in sorted(standard):
            self.voice_combo.addItem(f"🔊 {format_name(path)}", str(path))
        for path in sorted(cloned):
            self.voice_combo.addItem(f"👤 {format_name(path)} (Clonada)", str(path))

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Selecionar Textos", "", "Textos (*.txt)")
        for p in paths:
            self._files.append({"path": p, "voice": None, "lang": "en"})
        self._refresh_list()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Textos")
        if folder:
            paths = sorted([str(p) for p in Path(folder).glob("*.txt")])
            for p in paths:
                self._files.append({"path": p, "voice": None, "lang": "en"})
            self._refresh_list()

    def _clear_files(self):
        self._files.clear()
        self._refresh_list()

    def _refresh_list(self):
        self.files_list.clear()
        for i, d in enumerate(self._files):
            name = Path(d['path']).name
            vox = Path(d['voice']).stem if d['voice'] else "Padrão"
            lang = d.get('lang', 'en')
            self.files_list.addItem(f"#{i+1} {name} | Voz: {vox} | {lang}")

    def _apply_voice(self):
        idx = self.files_list.currentRow()
        if idx >= 0:
            vData = self.voice_combo.currentData()
            self._files[idx]['voice'] = vData
            self._files[idx]['lang'] = self.lang_combo.currentText()
            self._refresh_list()

    def resolve_voice_data(self, voice_data: Optional[str]) -> Optional[str]:
        return voice_data

    def _get_tts_config(self):
        main_win = self.window()
        if hasattr(main_win, "tts_tab"):
            return main_win.tts_tab.get_session()
        return {}

    def _start_normal(self):
        self._run_all_mode = False
        self._start_pipeline(self._files)

    def _start_run_all(self):
        if not self._files:
            QMessageBox.warning(self, "Sem Arquivos", "Adicione pelo menos um .txt")
            return
        main_win = self.window()
        if hasattr(main_win, "images_tab") and not main_win.images_tab.get_images():
            QMessageBox.warning(self, "Sem Imagens", "Vá na aba Imagens e adicione imagens antes do Run All.")
            return
        reply = QMessageBox.question(self, "Run All", "Isso irá gerar áudio e vídeo para TODOS os arquivos .txt sequencialmente. Deseja continuar?")
        if reply != QMessageBox.StandardButton.Yes:
           return
        self._run_all_mode = True
        self._run_all_index = 0
        self._run_all_next()

    def _run_all_next(self):
        if self._run_all_index >= len(self._files):
            self._run_all_mode = False
            _append_log(self.log_text, "\\n🎉 Run All concluído com sucesso para todos os arquivos!")
            QMessageBox.information(self, "Run All Concluído", "Todos os projetos foram gerados.")
            return
        file_cfg = self._files[self._run_all_index]
        _append_log(self.log_text, f"\\n=== RUN ALL: Iniciando {Path(file_cfg['path']).name} ===")
        proj_base = self.project_edit.text().strip() or "projeto"
        if len(self._files) > 1:
            proj_name = f"{proj_base}_{self._run_all_index + 1}"
        else:
            proj_name = proj_base
        self._current_run_all_project = proj_name
        self._start_pipeline([file_cfg], project_override=proj_name)

    def _start_pipeline(self, configs, project_override=None):
        if not configs: return
        self.btn_generate.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._generated.clear()
        self.audio_list.clear()
        cfg = self._get_tts_config()
        self._pipeline = AudioPipeline(
            file_configs=configs,
            project_name=project_override or (self.project_edit.text().strip() or "projeto"),
            output_root=self.output_root_edit.text().strip() or "output",
            model_type=cfg.get("model_type", "turbo"),
            whisper_model=cfg.get("whisper_model", "base"),
            similarity_threshold=cfg.get("similarity_threshold", 0.75),
            max_retries=cfg.get("max_retries", 3),
            temperature=cfg.get("temperature", 0.8),
            exaggeration=cfg.get("exaggeration", 0.5),
            cfg_weight=cfg.get("cfg_weight", 0.5),
            seed=cfg.get("seed", 0)
        )
        self._worker_thread = QThread(self)
        self._pipeline.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._pipeline.run)
        self._pipeline.log_message.connect(self._on_log)
        self._pipeline.progress.connect(self._on_progress)
        self._pipeline.paragraph_done.connect(self._on_pdone)
        self._pipeline.finished.connect(self._on_done)
        self._worker_thread.start()

    @Slot(str)
    def _on_log(self, msg: str):
        _append_log(self.log_text, msg)

    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(current * 100 / total))
            self.progress_bar.setFormat(f"Parágrafo {current}/{total}")

    @Slot(int, str, str)
    def _on_pdone(self, index: int, wav_path: str, text: str):
        self._generated.append((index, wav_path, text))
        short = text[:40] + ("…" if len(text) > 40 else "")
        item = QListWidgetItem(f"#{index}  {Path(wav_path).name}  |  {short}")
        item.setData(Qt.ItemDataRole.UserRole, wav_path)
        self.audio_list.addItem(item)

    @Slot(bool, str)
    def _on_done(self, success: bool, message: str):
        self._worker_thread.quit()
        self._worker_thread.wait()
        if success:
            self.progress_bar.setValue(100)
            paths = [g[1] for g in self._generated]
            if self._run_all_mode:
                _append_log(self.log_text, f"✓ Áudio pronto. Passando para Geração de Vídeo...")
                self.run_all_audio_done.emit(self._run_all_index, self._current_run_all_project, paths)
            else:
                self.btn_generate.setEnabled(True)
                self.btn_run_all.setEnabled(True)
                self.btn_cancel.setEnabled(False)
                self.audio_generated.emit(paths)
                _append_log(self.log_text, f"✓ {message}")
        else:
            self.btn_generate.setEnabled(True)
            self.btn_run_all.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self._run_all_mode = False
            _append_log(self.log_text, f"✗ {message}")

    def continue_run_all(self):
        if self._run_all_mode:
            self._run_all_index += 1
            self._run_all_next()

    def _cancel_generation(self):
        if self._pipeline:
            self._pipeline.cancel()
        self._run_all_mode = False
        self.btn_cancel.setEnabled(False)
        _append_log(self.log_text, "⚠ Cancelamento solicitado…")

    def _play_selected(self):
        row = self.audio_list.currentRow()
        if row < 0: return
        wav = self.audio_list.item(row).data(Qt.ItemDataRole.UserRole)
        self._play_wav(wav, row)

    def _play_wav(self, wav: str, row: Optional[int] = None):
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(wav))
        self._player.play()
        self._playing_row = row

    def _stop_audio(self):
        self._chain_active = False
        self._player.stop()

    def _play_chain(self):
        if self.audio_list.count() == 0: return
        self._chain_active = True
        self._chain_index = 0
        self._play_chain_next()

    def _play_chain_next(self):
        if not self._chain_active or self._chain_index >= self.audio_list.count():
            self._chain_active = False
            return
        item = self.audio_list.item(self._chain_index)
        self.audio_list.setCurrentRow(self._chain_index)
        wav = item.data(Qt.ItemDataRole.UserRole)
        self._play_wav(wav, self._chain_index)

    @Slot()
    def _on_playback_state(self):
        from PySide6.QtMultimedia import QMediaPlayer as _QMP
        if self._player.playbackState() == _QMP.PlaybackState.StoppedState:
            if self._chain_active:
                self._chain_index += 1
                self._play_chain_next()

    def _regen_selected(self):
        row = self.audio_list.currentRow()
        if row < 0 or row >= len(self._generated):
            QMessageBox.information(self, "Selecione", "Selecione um áudio na lista.")
            return
        idx, wav_path, text = self._generated[row]
        project = self.project_edit.text().strip() or "projeto"
        import tempfile, textwrap
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
            tmp_txt = f.name
        row_voice = None
        row_lang = "en"
        val = self._files[0] if self._files else {}
        if row < len(self._files): val = self._files[row]
        vData = val.get("voice")
        if vData: row_voice = vData
        row_lang = val.get("lang", "en")
        cfg = self._get_tts_config()
        pipeline = AudioPipeline(
            file_configs=[{"path": tmp_txt, "voice": row_voice, "lang": row_lang}],
            project_name=project,
            output_root=self.output_root_edit.text().strip() or "output",
            model_type=cfg.get("model_type", "turbo"),
            whisper_model=cfg.get("whisper_model", "base"),
            similarity_threshold=cfg.get("similarity_threshold", 0.75),
            max_retries=cfg.get("max_retries", 3),
            temperature=cfg.get("temperature", 0.8),
            exaggeration=cfg.get("exaggeration", 0.5),
            cfg_weight=cfg.get("cfg_weight", 0.5),
            seed=cfg.get("seed", 0)
        )
        thread = QThread(self)
        pipeline.moveToThread(thread)
        def _on_regen_done(success, msg):
            Path(tmp_txt).unlink(missing_ok=True)
            thread.quit()
            new_wav = (Path(self.output_root_edit.text().strip() or "output") / project / "audios" / "audio_1.wav")
            if success and new_wav.exists():
                import shutil
                shutil.copy2(str(new_wav), str(wav_path))
                _append_log(self.log_text, f"✓ Áudio #{idx} regravado.")
            else:
                _append_log(self.log_text, f"✗ Falha ao regravar #{idx}.")
        thread.started.connect(pipeline.run)
        pipeline.log_message.connect(self._on_log)
        pipeline.finished.connect(_on_regen_done)
        _append_log(self.log_text, f"↻ Regravando parágrafo #{idx}…")
        thread.start()

    def _open_output_folder(self):
        project = self.project_edit.text().strip() or "projeto"
        folder = Path(self.output_root_edit.text().strip() or "output") / project
        import subprocess as _sp
        if folder.exists():
            _sp.Popen(["explorer", str(folder.resolve())])

    def get_generated_paths(self) -> List[str]:
        return [g[1] for g in self._generated]

    def load_session(self, data: dict):
        if "project" in data: self.project_edit.setText(data["project"])
        if "output_root" in data: self.output_root_edit.setText(data["output_root"])

    def get_session(self) -> dict:
        return {
            "project": self.project_edit.text().strip(),
            "output_root": self.output_root_edit.text().strip()
        }

"""

with open("c:/Users/desot/Downloads/Chatterbox-TTS-Server-main/manhwa_app/app.py", "r", encoding="utf-8") as f:
    text = f.read()

start_idx = text.find("class AudioTab(QWidget):")
end_idx = text.find("# Aba 2 — Imagens")

if start_idx != -1 and end_idx != -1:
    new_text = text[:start_idx] + NEW_AUDIO_TAB + text[end_idx-42:]
    with open("c:/Users/desot/Downloads/Chatterbox-TTS-Server-main/manhwa_app/app.py", "w", encoding="utf-8") as f:
        f.write(new_text)
    print("PATCH SUCESSO!")
else:
    print("FALHA AO ENCONTRAR INDICES")
