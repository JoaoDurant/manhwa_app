import os
import sys
import time
import json
from google import genai
from google.genai import types

def test_model(client, model_name, log_file):
    msg = f"--- Testando {model_name} ---\n"
    print(msg, end="")
    log_file.write(msg)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Diga 'Olá' para teste.",
        )
        res = f"Sucesso: {response.text}\n"
        print(res, end="")
        log_file.write(res)
        return True
    except Exception as e:
        err = f"Erro em {model_name}: {e}\n"
        print(err, end="")
        log_file.write(err)
        return False

if __name__ == "__main__":
    api_key = None
    if os.path.exists("session_config.json"):
        try:
            with open("session_config.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                gemini_cfg = data.get("settings_tab", {}).get("gemini", {})
                api_key = gemini_cfg.get("api_key")
        except: pass

    with open("gemini_results.log", "w", encoding="utf-8") as log:
        if not api_key:
            log.write("API Key não encontrada no session_config.json.\n")
            sys.exit(1)

        client = genai.Client(api_key=api_key)
        # Testar os que apareceram no list()
        models_to_test = [
            "gemini-3.1-pro-preview",
            "gemini-flash-latest",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash"
        ]
        
        for m in models_to_test:
            test_model(client, m, log)
            time.sleep(3)
