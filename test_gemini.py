import json
import logging
import os
import time
from pathlib import Path
import google.genai as genai
from google.genai import types

from gemini_processor import GeminiProcessor

logging.basicConfig(level=logging.DEBUG, format='[%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("TEST_GEMINI")

def find_api_key():
    config_file = Path("session_config.json")
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text("utf-8"))
            return data.get("gemini", {}).get("api_key", "")
        except:
            pass
    return ""

def standalone_test():
    api_key = find_api_key()
    if not api_key:
        logger.error("Nenhuma chava API encontrada em session_config.json. Crie ou passe manualmente no console.")
        api_key = input("Cole sua Gemini API Key: ").strip()
        
    if not api_key:
        logger.error("Sem API Key. Teste abortado.")
        return

    # Modelos para testar
    models_to_test = [
        "gemini-2.5-flash", 
        "gemini-2.5-pro", 
        "gemini-3.0-flash-preview", 
        "gemini-2.0-flash"
    ]
    
    valid_models = []
    
    # Texto e Prompt Customizado
    test_text = "1. O herói sacou sua espada prateada sob a luz da lua.\n2. O monstro recuou, emitindo um rosnado gutural."
    prompt = f"Traduza e revise isso num formato JSON válido exigido: {test_text}"
    logger.info(f"== Iniciando Teste Isolado de Modelos Gemini ==")
    
    processor = GeminiProcessor()
    
    for model in models_to_test:
        logger.info(f"--- Testando Modelo: {model} ---")
        processor.model_name = model
        
        # Validando Inicialização e Key
        try:
            processor._client = genai.Client(api_key=api_key)
        except Exception as e:
            logger.error(f"Falha ao inicializar o Client para {model}: {e}")
            continue
            
        # Simulação direta sem _call_with_retry para pegar o raw exato exigido pela Etapa 1
        config_kwargs = {"response_mime_type": "application/json"}
        if "thinking" in model.lower() or "3" in model:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="high")
            
        logger.info(f"Fazendo request...")
        
        start_t = time.time()
        try:
            response = processor._client.models.generate_content(
                model=model,
                contents=[{"parts": [{"text": prompt}]}],
                config=types.GenerateContentConfig(**config_kwargs)
            )
            elapsed = time.time() - start_t
            
            # Status e Resposta
            logger.info(f"Tempo de requisição: {elapsed:.2f}s")
            logger.info(f"Tipo da resposta: {type(response)}")
            
            # Etapa 2 Normalizer Test
            try:
                raw_extracted = processor.normalize_gemini_response(response)
                logger.info(f"JSON Normalizado com Sucesso! Tamanho: {len(raw_extracted)} chars")
                if len(raw_extracted) > 0:
                    valid_models.append(model)
                    logger.debug(f"Conteúdo Bruto Extraído: {raw_extracted}")
                else:
                    logger.error("Resposta devolveu formato vazio válido, mas sem conteúdo.")
            except Exception as norm_e:
                logger.error(f"Falha de normalização: {norm_e}")
                
        except Exception as api_exc:
            err_str = str(api_exc)
            logger.error(f"Erro na API (Status / Exceção): {err_str[:150]}")
            if "400" in err_str:
                logger.error("⚠️ Isso é um erro 400. Sua API KEY pode ser inválida ou o modelo não suporta o parâmetro enviado.")
            elif "429" in err_str:
                logger.warning("⚠️ Erro 429 de Quota! Pausando por 5 segundos...")
                time.sleep(5)
                
        logger.info("-----------------------------")

    logger.info(f"== TESTE CONCLUÍDO ==")
    logger.info(f"Modelos Funcionais Validados ({len(valid_models)}): {valid_models}")

if __name__ == "__main__":
    standalone_test()
