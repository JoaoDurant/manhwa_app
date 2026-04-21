import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from manhwa_app.dashboard_timing import DashboardTiming

dt = DashboardTiming()

# --- Clock 1: Audio paragraph timer ---
dt.start_paragraph(idx=3, total=47)
time.sleep(0.05)
elapsed = dt.get_paragraph_elapsed()
assert 0.04 < elapsed < 0.15, f"[FAIL] paragraph elapsed fora do range: {elapsed}"
print(f"  [PASS] Clock1 paragraph elapsed: {elapsed:.3f}s ✓")

# --- Clock 2: Job timer + ETA ---
dt.start_job(total_paragraphs=47)
# Simula 10 parágrafos completados em 43s
for i in range(10):
    dt.record_paragraph_complete(elapsed_s=4.3)

job_elapsed = dt.get_job_elapsed()
job_eta = dt.get_job_eta()
assert job_eta > 0, f"[FAIL] job_eta negativo ou zero: {job_eta}"
assert job_eta < 300, f"[FAIL] job_eta improvável: {job_eta}s"
print(f"  [PASS] Clock2 job_eta: {job_eta:.1f}s ✓")

# --- ETA não calculado antes de 3 parágrafos ---
dt2 = DashboardTiming()
dt2.start_job(total_paragraphs=47)
dt2.record_paragraph_complete(elapsed_s=4.0)
dt2.record_paragraph_complete(elapsed_s=4.0)
eta_early = dt2.get_job_eta()
assert eta_early is None, f"[FAIL] ETA deveria ser None antes de 3 paras: {eta_early}"
print("  [PASS] ETA retorna None antes de 3 parágrafos ✓")

# --- Clock 3: Queue ETA ---
dt.start_queue(total_jobs=5)
dt.record_job_complete(elapsed_s=260.0)
dt.record_job_complete(elapsed_s=280.0)
queue_eta = dt.get_queue_eta(remaining_jobs=3)
assert 700 < queue_eta < 900, f"[FAIL] queue_eta improvável: {queue_eta}s"
print(f"  [PASS] Clock3 queue_eta: {queue_eta:.1f}s ✓")

# --- Nunca retorna negativo ---
dt3 = DashboardTiming()
dt3.start_job(total_paragraphs=5)
for _ in range(5):
    dt3.record_paragraph_complete(elapsed_s=1.0)
# Todos completos — ETA deve ser 0, nunca negativo
eta_final = dt3.get_job_eta()
assert eta_final is not None and eta_final >= 0, \
    f"[FAIL] ETA negativo no final: {eta_final}"
print(f"  [PASS] ETA nunca negativo: {eta_final:.1f}s ✓")

print("\n[ALL TIMING TESTS OK]")
