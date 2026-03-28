import json
import logging
import unittest
from gemini_processor import GeminiProcessor

logging.basicConfig(level=logging.INFO)

class DummyContent:
    def __init__(self, text):
        self.parts = [type('Part', (), {'text': text})()]

class DummyCandidate:
    def __init__(self, text):
        self.content = DummyContent(text)

class DummyResponse:
    def __init__(self, text):
        self.text = text
        self.candidates = [DummyCandidate(text)]

class TestGeminiProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = GeminiProcessor()

    def test_normalize_string(self):
        res = " simple string "
        self.assertEqual(self.processor.normalize_gemini_response(res), "simple string")

    def test_normalize_dict(self):
        res = {"paragrafos": ["um", "dois"]}
        # without candidates, should dumps to string
        normalized = self.processor.normalize_gemini_response(res)
        self.assertIn("paragrafos", normalized)
        self.assertIn("um", normalized)

    def test_normalize_dict_with_candidates(self):
        # Format matching genai JSON mode structure if returned as raw dict
        res = {"candidates": [{"content": {"parts": [{"text": "{\"paragrafos\": [\"1\"]}"}]}}]}
        normalized = self.processor.normalize_gemini_response(res)
        self.assertEqual(normalized, '{"paragrafos": ["1"]}')

    def test_normalize_list(self):
        res = [{"paragrafos": ["a", "b"]}]
        normalized = self.processor.normalize_gemini_response(res)
        self.assertIn("paragrafos", normalized)

    def test_normalize_dummy_SDK_response(self):
        res = DummyResponse('{"paragrafos": ["texto"]}')
        normalized = self.processor.normalize_gemini_response(res)
        self.assertEqual(normalized, '{"paragrafos": ["texto"]}')

    def test_validate_output_success(self):
        raw = "1. Texto um\n2. Texto dois\n3. Texto três"
        lines = self.processor.validate_output(raw, 3)
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "Texto um")
        self.assertEqual(lines[2], "Texto três")

    def test_validate_output_fail(self):
        raw = "1. Aqui só tem um"
        with self.assertRaises(ValueError):
            self.processor.validate_output(raw, 2)
            
    def test_retry_extracts_list_protection(self):
        # Extract json method test with array root
        raw = '[{"paragrafos": ["texto 1"]}]'
        extracted = self.processor._extract_json(raw)
        data = json.loads(extracted)
        
        # This mirrors _call_with_retry logic
        if isinstance(data, list):
            data = data[0]
            
        self.assertTrue(isinstance(data, dict))
        self.assertEqual(data.get("paragrafos")[0], "texto 1")

if __name__ == '__main__':
    unittest.main()
