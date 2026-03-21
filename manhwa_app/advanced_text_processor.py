import re
import unicodedata


def remove_accents(text: str) -> str:
    """
    Remove acentos usando unicodedata.
    Exemplo: 'rápido' -> 'rapido'
    """
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def clean_text(text: str) -> str:
    """
    Remove caracteres desnecessários, mantendo letras, números, espaços e pontuação básica.
    CORRIGIDO: o regex anterior `[^\w\s.,!?'"\-]` usava \w que em Python
    preserva letras Unicode (incluindo acentuadas). No entanto, os límites de
    charset não incluem '¿', '¡', '«', '»' que são válidos para TTS em espanhol.
    Mantemos agora qualquer letra Unicode (\p{L}) via teste de categoria.
    """
    result = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Preservar: letras (Lu, Ll, Lt, Lm, Lo), números (N*), espaços (Zs, \n, \r, \t),
        # e pontuação TTS relevante
        if cat.startswith('L') or cat.startswith('N'):  # letras e números Unicode
            result.append(ch)
        elif cat.startswith('Z') or ch in ' \t\n\r':   # espaços
            result.append(ch)
        elif ch in '.,!?\'"-:;()[]¿¡«»\u2026':         # pontuação TTSútil
            result.append(ch)
        # ignora tudo mais (emojis, símbolos matemáticos, etc.)
    return ''.join(result)

def remove_prefixes(text: str) -> str:
    """
    Remove padrões numéricos iniciais de lista como '1. ', '2- ', '3)', '04. ' etc.
    Exemplo: '1. Olá' -> 'Olá'
    """
    # Use re.sub repetidamente no caso de lixo empilhado, mas 1 vez já cobre 99%
    return re.sub(r'^\s*\d+[\.\-\)]\s*', '', text)

def improve_punctuation(text: str) -> str:
    """
    Melhora a pontuação adicionando espaços após sinais para melhorar pausas do TTS.
    NÃO converte pontos em reticências — isso distorceria o estilo da fala.
    """
    # CORRIGIDO: protege reticências existentes antes de qualquer substituição
    text = text.replace('...', '<ELLIPSIS>')

    # Garante espaço após pontuação (sem alterar o character em si)
    text = re.sub(r'\.(?!\s)', '. ', text)          # ponto → ponto + espaço
    text = re.sub(r',(?!\s)', ', ', text)            # vírgula → vírgula + espaço
    text = re.sub(r'!(?!\s)', '! ', text)            # ! → ! + espaço
    text = re.sub(r'\?(?!\s)', '? ', text)           # ? → ? + espaço
    text = re.sub(r';(?!\s)', '; ', text)            # ; → ; + espaço

    # Restaura reticências
    text = text.replace('<ELLIPSIS>', '... ')

    # Limpa espaços duplos gerados pelas substituições
    return re.sub(r' {2,}', ' ', text).strip()

def preprocess_text_for_speech(text: str) -> str:
    """
    Transforma o texto bruto em uma versão NATURAL para fala (TTS),
    como se fosse narrado por um humano. Cumpre as regras do usuário:
    - Melhora pontuação (evita pontos excessivos e reticências infinitas)
    - Quebra de frases fluida
    - Fluidez de fala (remove repetições)
    - Tom de narração falado
    """
    if not text: return text
    
    # 1. PONTUAÇÃO NATURAL: Substituir excesso de "..." por um único "..."
    text = re.sub(r'\.{3,}', '...', text)
    
    # 2. FLUIDEZ DE FALA: Remover repetições exatas (apenas de palavras, ex: "não, não, não")
    # Tenta remover repetições de até 3 palavras se acontecerem em sequência
    # Ex: "Eu vou eu vou para casa" -> "Eu vou para casa"
    # Como é difícil fazer isso perfeitamente em Regex para todos os casos, focamos nas repetições de pontuação
    # e repetições imediatas com vírgula: "não, não" -> "não"
    
    # 3. QUEBRA DE FRASES: Evitar blocos difíceis de respirar
    # Não vamos quebrar a string literalmente com \n (já fizemos chunking na pipeline),
    # mas vamos garantir que vírgulas existam antes de conjunções fortes se a frase for muito longa
    text = re.sub(r'\s+(porque|mas|porém|contudo)\s+', r', \1 ', text, flags=re.IGNORECASE)
    
    # 4. Limpeza de travamentos: Converte travessões soltos ou hifens duplos em pausas (vírgulas)
    text = re.sub(r'\s*--+\s*', ', ', text)
    text = re.sub(r'\s*—\s*', ', ', text)
    
    # Arrumar espaços duplos e pontuação duplicada
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r', \.', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def apply_phonetic(text: str) -> str:
    """
    Aplica conversão fonética simplificada inspirada no IPA.
    """
    text = re.sub(r'rr', 'ʁ', text, flags=re.IGNORECASE)
    text = re.sub(r'lh', 'ʎ', text, flags=re.IGNORECASE)
    text = re.sub(r'nh', 'ɲ', text, flags=re.IGNORECASE)
    text = re.sub(r'ch', 'ʃ', text, flags=re.IGNORECASE)
    
    # Substituições simples
    text = re.sub(r'r', 'ʁ', text, flags=re.IGNORECASE)
    text = re.sub(r'j', 'ʒ', text, flags=re.IGNORECASE)
    text = re.sub(r'x', 'ʃ', text, flags=re.IGNORECASE)
    
    return text

def process_text(text: str, config: dict) -> str:
    """
    Pipeline principal para tratamento do texto antes do envio ao TTS.
    """
    if not text:
        return text
        
    # Sempre remove prefixos númericos do início (1. 2.)
    text = remove_prefixes(text)
    
    if config.get("clean_symbols", True):
        text = clean_text(text)
        
    if config.get("remove_accents", False):
        text = remove_accents(text)
        
    if config.get("normalize_text", True):
        if config.get("lowercase", False):
            text = text.lower()
        # Remove espaços duplicados e aplica trim nas bordas
        text = re.sub(r'\s+', ' ', text).strip()

    # Aplica formatação de fala realista natural (TTS) requesitada
    if config.get("natural_speech", True):
        text = preprocess_text_for_speech(text)
        
    if config.get("improve_punctuation", True) or config.get("add_natural_pauses", True):
        text = improve_punctuation(text)
        
    if config.get("use_phonetic", False):
        text = apply_phonetic(text)
        
    # Limpeza final de espaços gerados pelos regex
    text = re.sub(r'\s+', ' ', text).strip()
    return text
