from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

class QueueTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 9, parent)
        self.jobs = {}
        
    def add_job(self, job_id, project, workflow, engine, total_paragraphs):
        row = self.rowCount()
        self.insertRow(row)
        self.jobs[job_id] = {
            "row": row,
            "status": "Wait",
            "audio_pct": 0,
            "video_pct": 0
        }
        
    def set_job_status(self, job_id, status):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            
    def update_audio_progress(self, job_id, done, total):
        if job_id in self.jobs and total > 0:
            self.jobs[job_id]["audio_pct"] = (done / total) * 100
            
    def update_video_progress(self, job_id, done, total):
        if job_id in self.jobs and total > 0:
            self.jobs[job_id]["video_pct"] = (done / total) * 100
            
    def get_status_text(self, job_id):
        return self.jobs.get(job_id, {}).get("status", "")
        
    def get_audio_progress_pct(self, job_id):
        return self.jobs.get(job_id, {}).get("audio_pct", 0)
