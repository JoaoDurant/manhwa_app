import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

from manhwa_app.ui.log_widget import LogWidget  # ajuste o import se necessário

log = LogWidget(max_lines=100)
log.show()

# Teste 1: Adiciona entradas de todos os tipos
log.append_entry(stage="AUDIO", level="INFO",  message="Para 1/47 aceito")
log.append_entry(stage="AUDIO", level="WARN",  message="Retry 2/3")
log.append_entry(stage="VIDEO", level="INFO",  message="Composição iniciada")
log.append_entry(stage="MACRO", level="ERROR", message="Job falhou")
log.append_entry(stage="MACRO", level="SUCCESS", message="Queue completa")

app.processEvents()

content = log.toPlainText() if hasattr(log, "toPlainText") else log.get_content()
assert "Para 1/47" in content, "[FAIL] Entrada INFO não encontrada no log"
assert "Retry 2/3" in content, "[FAIL] Entrada WARN não encontrada"
print("  [PASS] Entradas adicionadas corretamente ✓")

# Teste 2: Limite de linhas (circular buffer)
log2 = LogWidget(max_lines=10)
for i in range(25):
    log2.append_entry("AUDIO", "INFO", f"Linha {i}")
app.processEvents()
line_count = log2.document().lineCount() if hasattr(log2, "document") else len(log2.get_lines())
assert line_count <= 12, f"[FAIL] Log excedeu limite: {line_count} linhas"
print(f"  [PASS] Circular buffer OK: {line_count} linhas ✓")

# Teste 3: Filtro "Errors Only"
log3 = LogWidget(max_lines=100)
log3.append_entry("AUDIO", "INFO",  "Info normal")
log3.append_entry("AUDIO", "ERROR", "Erro crítico")
log3.set_filter("errors_only")
app.processEvents()
visible = log3.get_visible_content()
assert "Erro crítico" in visible, "[FAIL] Erro não visível com filtro errors_only"
assert "Info normal" not in visible, "[FAIL] INFO visível com filtro errors_only"
print("  [PASS] Filtro errors_only OK ✓")

print("\n[ALL LOG TESTS OK]")
