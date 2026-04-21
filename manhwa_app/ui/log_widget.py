from PySide6.QtWidgets import QWidget, QTextEdit, QVBoxLayout

class LogWidget(QWidget):
    def __init__(self, max_lines=100, parent=None):
        super().__init__(parent)
        self.max_lines = max_lines
        self.lines = []
        self.filter = "all"
        
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

    def append_entry(self, stage, level, message):
        entry = {"stage": stage, "level": level, "message": message, "text": f"[{stage}] [{level}] {message}"}
        self.lines.append(entry)
        if len(self.lines) > self.max_lines:
            self.lines.pop(0)

    def toPlainText(self):
        return "\n".join([line["text"] for line in self.lines])

    def get_lines(self):
        return self.lines

    def set_filter(self, filter_name):
        self.filter = filter_name

    def get_visible_content(self):
        visible = []
        for line in self.lines:
            if self.filter == "errors_only" and line["level"] != "ERROR":
                continue
            visible.append(line["text"])
        return "\n".join(visible)
