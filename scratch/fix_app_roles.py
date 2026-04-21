"""
Fix DisplayRole+1 (=DecorationRole, icon role=1) -> UserRole+1 (=257, safe custom slot)
in manhwa_app/app.py handlers _on_psynthesized and _on_pdone.
"""
import re

path = r'e:\backup\v6\manhwa_app\app.py'
with open(path, encoding='utf-8') as f:
    content = f.read()

original = content

# ---- Fix 1: _on_psynthesized setData ----
old1 = '        item.setData(Qt.ItemDataRole.DisplayRole + 1, index) # Store index for easy lookup'
new1 = (
    '        # [FIX] DisplayRole+1 = DecorationRole (Qt icon role=1, unsafe for user data).\n'
    '        # UserRole+1 = 257 is a safe custom slot Qt never overwrites.\n'
    '        item.setData(Qt.ItemDataRole.UserRole + 1, index)'
)
assert old1 in content, "Fix1 target not found"
content = content.replace(old1, new1, 1)
print("Fix 1 applied: _on_psynthesized setData")

# ---- Fix 2: _on_pdone search comparison ----
old2 = '            if item.data(Qt.ItemDataRole.DisplayRole + 1) == index:'
new2 = '            # [FIX] Was DisplayRole+1 (=DecorationRole=1), unsafe. UserRole+1=257 is safe.\n            if item.data(Qt.ItemDataRole.UserRole + 1) == index:'
assert old2 in content, "Fix2 target not found"
content = content.replace(old2, new2, 1)
print("Fix 2 applied: _on_pdone comparison")

# ---- Fix 3: _on_pdone fallback - add UserRole+1 to fallback item ----
old3 = (
    '        # Fallback caso n\u00e3o encontre (n\u00e3o deveria acontecer)\n'
    '        item = QListWidgetItem(f"#{index}  \u2705 {Path(wav_path).name}  |  {short}")\n'
    '        item.setData(Qt.ItemDataRole.UserRole, wav_path)\n'
    '        self.audio_list.addItem(item)'
)
new3 = (
    '        # Fallback: item not found (Smart Fill existing audio) \u2014 add directly\n'
    '        item = QListWidgetItem(f"#{index}  \u2705 {Path(wav_path).name}  |  {short}")\n'
    '        item.setData(Qt.ItemDataRole.UserRole, wav_path)\n'
    '        item.setData(Qt.ItemDataRole.UserRole + 1, index)\n'
    '        self.audio_list.addItem(item)'
)
assert old3 in content, "Fix3 target not found"
content = content.replace(old3, new3, 1)
print("Fix 3 applied: _on_pdone fallback item")

# ---- Also add UserRole+1 to the UPDATE path inside _on_pdone ----
old4 = (
    '                item.setData(Qt.ItemDataRole.UserRole, wav_path)\n'
    '                return'
)
new4 = (
    '                item.setData(Qt.ItemDataRole.UserRole, wav_path)\n'
    '                item.setData(Qt.ItemDataRole.UserRole + 1, index)  # keep consistent\n'
    '                return'
)
assert old4 in content, "Fix4 target not found"
content = content.replace(old4, new4, 1)
print("Fix 4 applied: _on_pdone update path keeps UserRole+1")

assert content != original, "No changes were made!"

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\nAll fixes applied and saved.")
