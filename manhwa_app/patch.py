import re

app_path = "e:/backup/v6/manhwa_app/app.py"
new_tab_path = "e:/backup/v6/manhwa_app/new_macro_tab_class.py"

with open(app_path, "r", encoding="utf-8") as f:
    text = f.read()

with open(new_tab_path, "r", encoding="utf-8") as f:
    new_macro_tab = f.read()

# Find MacroTab class boundaries
start_txt = "class MacroTab(QWidget):"
end_txt = "class MainWindow(QMainWindow):"

start_idx = text.find(start_txt)
end_idx = text.find(end_txt, start_idx)

if start_idx == -1 or end_idx == -1:
    print("Could not find MacroTab or MainWindow boundaries!")
    exit(1)

# Find the end of imports, we need to inject the import for DashboardTiming
import_idx = text.find("from manhwa_app.utils import _append_log, natural_sort_key")
if import_idx != -1:
    text = text[:import_idx] + "from manhwa_app.dashboard_timing import DashboardTiming\n" + text[import_idx:]
    # adjust indices
    start_idx = text.find(start_txt)
    end_idx = text.find(end_txt, start_idx)

# Find the line before MainWindow to preserve comments
# Let's just back up to the nearest '#' line right before MainWindow, if any, or just blank lines.
while text[end_idx-1] in (' ', '\n', '\t'):
    end_idx -= 1

# If there is a "--- MainWindow ---" comment, back up before it
comment_idx = text.rfind("# ---------------------------------------------------------------------------", start_idx, end_idx)
if comment_idx != -1:
    end_idx = comment_idx

# Note: the new_macro_tab already has the import inside it at the top, we should strip it
new_macro_tab_real = re.sub(r'from manhwa_app\.dashboard_timing import DashboardTiming\n+', '', new_macro_tab)


new_text = text[:start_idx] + new_macro_tab_real + "\n\n" + text[end_idx:]

with open(app_path, "w", encoding="utf-8") as f:
    f.write(new_text)

print("Patch applied successfully.")
