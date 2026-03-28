# tests/test_json_extraction.py
import sys
from pathlib import Path

# Add root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from gemini_processor import GeminiProcessor

def test_robust_json_extraction():
    processor = GeminiProcessor()
    
    # Case 1: Pure JSON
    raw1 = '{"paragrafos": ["test1", "test2"]}'
    assert processor._extract_json(raw1) == raw1
    
    # Case 2: Markdown wrapped
    raw2 = 'Sure! Here it is:\n```json\n{"paragrafos": ["test1", "test2"]}\n```\nHope it helps!'
    extracted2 = processor._extract_json(raw2)
    assert '{"paragrafos": ["test1", "test2"]}' in extracted2
    
    # Case 3: List instead of object
    raw3 = 'Some text before ["item1", "item2"] some text after'
    extracted3 = processor._extract_json(raw3)
    assert extracted3 == '["item1", "item2"]'

    print("Robust JSON extraction tests passed!")

if __name__ == "__main__":
    test_robust_json_extraction()
