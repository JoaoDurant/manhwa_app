import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

from manhwa_app.ui.queue_table import QueueTable  # ajuste se necessário

table = QueueTable()
table.show()

# Adiciona jobs
table.add_job(job_id="j1", project="arc_06", workflow="audio_video",
              engine="CB-ML", total_paragraphs=47)
table.add_job(job_id="j2", project="arc_07", workflow="audio_video",
              engine="CB-ML", total_paragraphs=52)

app.processEvents()
assert table.rowCount() == 2, f"[FAIL] Esperava 2 linhas, got {table.rowCount()}"
print("  [PASS] Jobs adicionados à tabela ✓")

# Atualiza status
table.set_job_status("j1", "running")
table.update_audio_progress("j1", done=23, total=47)
app.processEvents()

status_text = table.get_status_text("j1")
assert status_text in ("Running", "🔵 Run", "running"), \
    f"[FAIL] Status inesperado: {status_text}"
print("  [PASS] Status 'running' aplicado ✓")

audio_pct = table.get_audio_progress_pct("j1")
assert 48 <= audio_pct <= 50, f"[FAIL] Audio progress inesperado: {audio_pct}%"
print(f"  [PASS] Audio progress: {audio_pct}% ✓")

# Finaliza job
table.set_job_status("j1", "done")
table.update_audio_progress("j1", done=47, total=47)
table.update_video_progress("j1", done=47, total=47)
app.processEvents()

final_status = table.get_status_text("j1")
assert final_status in ("Done", "✅ Done", "done"), \
    f"[FAIL] Status final inesperado: {final_status}"
print("  [PASS] Job finalizado corretamente ✓")

print("\n[ALL QUEUE TABLE TESTS OK]")
