# manhwa_app/text_processor.py
import logging
import subprocess
import sys

import threading
logger = logging.getLogger(__name__)

# CORRIGIDO: cache por idioma (dict) em vez de singleton global
# Permite suporte a múltiplos idiomas no mesmo pipeline sem sobrescrever
_SPACY_MODELS: dict = {}   # {lang_key: nlp_model}
_SPACY_LOCK = threading.Lock()

_LANG_MODEL_MAP = {
    "pt": "pt_core_news_sm",
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "ja": "ja_core_news_sm",
    "zh": "zh_core_web_sm",
    "ko": "ko_core_news_sm",
}

def _get_model_name(lang: str) -> str:
    prefix = lang[:2].lower()
    return _LANG_MODEL_MAP.get(prefix, "en_core_web_sm")


def init_spacy(lang: str = "pt"):
    """
    Inicializa o spaCy para um idioma específico e armazena no cache por idioma.
    Garante que modelos de idiomas diferentes não se sobrescrevam.
    """
    model_name = _get_model_name(lang)
    with _SPACY_LOCK:
        if model_name in _SPACY_MODELS:
            return  # já carregado para esse idioma

        try:
            import spacy
        except ImportError:
            logger.warning("spaCy não está instalado. Execute: pip install spacy")
            return

        try:
            nlp = spacy.load(model_name)
            _SPACY_MODELS[model_name] = nlp
            logger.info(f"spaCy: modelo '{model_name}' carregado para idioma '{lang}'.")
        except OSError:
            logger.warning(f"Modelo spaCy '{model_name}' não encontrado. Baixando...")
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                nlp = spacy.load(model_name)
                _SPACY_MODELS[model_name] = nlp
                logger.info(f"spaCy: modelo '{model_name}' baixado e carregado.")
            except Exception as e:
                logger.error(f"Falha ao carregar/baixar spaCy '{model_name}': {e}")


def process_text_fluency(text: str, max_sentence_len: int = 250, lang: str = "pt") -> str:
    """
    Usa o spaCy para dividir frases muito longas em partes menores,
    inserindo pontuação onde apropriado para adicionar pausas naturais no TTS.
    """
    model_name = _get_model_name(lang)

    # Garante que o modelo está carregado para esse idioma
    if model_name not in _SPACY_MODELS:
        init_spacy(lang)

    nlp = _SPACY_MODELS.get(model_name)
    if nlp is None:
        return text  # Fallback seguro se não carregou

    doc = nlp(text)
    processed_sentences = []

    for sent in doc.sents:
        sent_str = sent.text.strip()
        if len(sent_str) <= max_sentence_len:
            processed_sentences.append(sent_str)
            continue

        chunks = []
        current_chunk = []
        current_len = 0

        for token in sent:
            current_chunk.append(token.text_with_ws)
            current_len += len(token.text_with_ws)

            if current_len > (max_sentence_len * 0.5):
                if token.pos_ == "PUNCT" and token.text in [",", ";", "-"]:
                    chunks.append("".join(current_chunk).strip())
                    current_chunk = []
                    current_len = 0
                elif token.pos_ in ["CCONJ", "SCONJ"]:
                    chunks.append("".join(current_chunk).strip())
                    current_chunk = []
                    current_len = 0

        if current_chunk:
            chunks.append("".join(current_chunk).strip())

        processed_sentences.extend(chunks)

    final_text = " ".join(processed_sentences)
    final_text = final_text.replace(" ,", ",")
    return final_text
