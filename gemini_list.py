import os
import json
from google import genai

if __name__ == "__main__":
    api_key = None
    if os.path.exists("session_config.json"):
        with open("session_config.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            api_key = data.get("settings_tab", {}).get("gemini", {}).get("api_key")

    if not api_key:
        print("Erro: API Key não encontrada.")
        exit(1)

    client = genai.Client(api_key=api_key)
    with open("gemini_models_utf8.txt", "w", encoding="utf-8") as out:
        out.write("--- Listando Modelos Disponíveis ---\n")
        try:
            for m in client.models.list():
                out.write(f"Model ID: {m.name}\n")
                if hasattr(m, 'input_token_limit'):
                    out.write(f"  input_token_limit: {m.input_token_limit}\n")
        except Exception as e:
            out.write(f"Erro ao listar modelos: {e}\n")
