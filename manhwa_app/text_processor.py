# manhwa_app/text_processor.py
import logging
import subprocess
import sys

import threading
logger = logging.getLogger(__name__)

_SPACY_NLP = None
_SPACY_READY = False
_SPACY_LOCK = threading.Lock()

def init_spacy(lang: str = "pt"):
    """
    Inicializa o spaCy e baixa o modelo se necessário.
    Garante que as dependências existam no runtime.
    """
    global _SPACY_NLP, _SPACY_READY
    with _SPACY_LOCK:
        if _SPACY_READY: return
        model_name = "pt_core_news_sm" if lang.startswith("pt") else "en_core_web_sm"
        
        try:
            import spacy
        except ImportError:
            logger.warning("spaCy não está instalado. Tentando instalar...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy", "spacy-lookups-data"])
                import spacy
            except Exception as e:
                logger.error(f"Falha ao instalar spaCy: {e}")
                return

        try:
            _SPACY_NLP = spacy.load(model_name)
            _SPACY_READY = True
            logger.info(f"spaCy: modelo '{model_name}' carregado.")
        except OSError:
            logger.warning(f"Modelo {model_name} não encontrado. Baixando...")
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                _SPACY_NLP = spacy.load(model_name)
                _SPACY_READY = True
                logger.info(f"spaCy: modelo '{model_name}' baixado e carregado.")
            except Exception as e:
                logger.error(f"Falha ao baixar {model_name}: {e}")
                return
            
    if _SPACY_READY:
        logger.info(f"spaCy: modelo '{model_name}' carregado.")

def process_text_fluency(text: str, max_sentence_len: int = 250, lang: str = "pt") -> str:
    """
    Usa o spaCy para dividir frases muito longas em partes menores,
    inserindo pontuação onde apropriado para adicionar pausas naturais no TTS.
    """
    global _SPACY_NLP, _SPACY_READY
    
    if not _SPACY_READY:
        init_spacy(lang)
    
    if not _SPACY_READY or not _SPACY_NLP:
        return text  # Fallback seguro
        
    doc = _SPACY_NLP(text)
    processed_sentences = []
    
    for sent in doc.sents:
        sent_str = sent.text.strip()
        # Se a frase já é curta o suficiente ou termina com pontuação forte
        if len(sent_str) <= max_sentence_len:
            processed_sentences.append(sent_str)
            continue
            
        # Para frases muito longas, tentamos dividir em conjunções ou pontuações fracas.
        # Procuramos tokens como vírgulas, 'e', 'mas', 'ou', 'que' onde podemos injetar reticências ou quebra
        chunks = []
        current_chunk = []
        current_len = 0
        
        for token in sent:
            current_chunk.append(token.text_with_ws)
            current_len += len(token.text_with_ws)
            
            if current_len > (max_sentence_len * 0.5):
                if token.pos_ == "PUNCT" and token.text in [",", ";", "-"]:
                    # Dividimos na vírgula sem injetar reticências para não quebrar o TTS
                    chunks.append("".join(current_chunk).strip())
                    current_chunk = []
                    current_len = 0
                elif token.pos_ in ["CCONJ", "SCONJ"]:
                    # Conjuncões (mas, e, que)
                    chunks.append("".join(current_chunk).strip())
                    current_chunk = []
                    current_len = 0

        if current_chunk:
            chunks.append("".join(current_chunk).strip())
            
        processed_sentences.extend(chunks)
        
    # Reune com espaços
    final_text = " ".join(processed_sentences)
    # Limpeza de espaços
    final_text = final_text.replace(" ,", ",")
    return final_text
