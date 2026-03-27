# gemini_processor.py
# Módulo de pré-processamento de roteiros via API Gemini.
# Roda ANTES do pipeline TTS: revisa e/ou traduz arquivos .txt numerados,
# salvando novos arquivos na mesma pasta do original.
# Nenhuma dependência do PySide6 — puro Python.

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 12   # parágrafos por chamada à API
OVERLAP = 2       # parágrafos de contexto (antes/depois), não entram no output

# Mapeamento de código de idioma → nome por extenso para o prompt
LANGUAGE_NAMES: dict[str, str] = {
    "en": "inglês americano",
    "es": "espanhol latino-americano",
    "fr": "francês",
    "de": "alemão",
    "it": "italiano",
    "ru": "russo",
    "ja": "japonês",
    "ko": "coreano",
    "zh": "chinês",
    "pt": "português brasileiro",
}

# ---------------------------------------------------------------------------
# Prompts do sistema
# ---------------------------------------------------------------------------

_REVISION_SYSTEM = """Você é um editor profissional especializado em roteiros de manhwa e webtoon narrados em áudio.

Sua tarefa: revisar APENAS os parágrafos do BLOCO ATUAL.

REGRAS OBRIGATÓRIAS:
1. Mantenha o sentido e a narrativa original — não reescreva, apenas corrija e melhore fluidez
2. Cada parágrafo deve terminar de forma que o próximo continue naturalmente (coesão narrativa)
3. Remova ou substitua: símbolos especiais (*, #, →, —), reticências excessivas (... ... ...), onomatopeias escritas (BOOM!, POW!), parênteses com indicações de cena
4. Escreva por extenso: números isolados vire palavras ("3" → "três"), siglas ambíguas explique ("km" → "quilômetros")
5. Frases muito longas (mais de 40 palavras): quebre em duas frases naturais
6. Não adicione frases novas — apenas refine o que existe
7. Mantenha o mesmo número de parágrafos que recebeu

CONTEXTO (não edite, apenas leia para entender a continuidade):
[ANTES]: {context_before}
[DEPOIS]: {context_after}

BLOCO ATUAL para revisar:
{current_block_formatted}

Responda SOMENTE com JSON válido, sem markdown, sem explicações:
{{"paragrafos": ["texto revisado 1", "texto revisado 2", ...]}}

O array deve ter exatamente {n} elementos, na mesma ordem dos parágrafos recebidos."""

_TRANSLATION_SYSTEM = """Você é um tradutor profissional especializado em manhwa e webtoon para narração em áudio.

Idioma de destino: {language_name}
Idioma de origem: {source_language_name}

REGRAS OBRIGATÓRIAS:
1. Traduza com naturalidade no idioma alvo — não traduza literalmente palavra por palavra
2. Mantenha o tom narrativo, a emoção e o ritmo do texto original
3. Nomes próprios de personagens: mantenha como estão (não traduza nomes)
4. A tradução deve soar como narração de áudio, não como texto escrito
5. Cada parágrafo deve fluir para o próximo (mesma coesão do original)
6. Mantenha o mesmo número de parágrafos

CONTEXTO em português (não traduza, só leia para entender a continuidade):
[ANTES]: {context_before}
[DEPOIS]: {context_after}

BLOCO ATUAL em português para traduzir:
{current_block_formatted}

Responda SOMENTE com JSON válido, sem markdown, sem explicações:
{{"paragrafos": ["texto traduzido 1", "texto traduzido 2", ...]}}

O array deve ter exatamente {n} elementos, na mesma ordem dos parágrafos recebidos."""


class GeminiProcessor:
    """Pre-processes numbered .txt scripts via the Gemini API.

    Revision and/or translation happens in a sliding-window chunking
    strategy with automatic cache, retries, and cancellation support.

    The *api_key* is NOT passed to ``__init__``; it is received by
    ``process()`` so the UI can control and validate it at runtime.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self._client = None  # inicializado em process() com a chave da UI

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def process(
        self,
        txt_path: str,
        api_key: str,
        languages: list[str],
        source_lang: str = "pt",
        delay_seconds: float = 4.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        stop_event=None,
        revision_prompt: str = None,
        translation_prompt: str = None,
        chunk_size: int = 12,
        overlap: int = 2,
        thinking_level: str = "high",
        media_resolution: str = "media_resolution_high",
    ) -> dict[str, str]:
        """Revise and optionally translate a numbered .txt file.

        Args:
            txt_path: Absolute path to the source .txt file.
            api_key: Gemini API key collected from the UI.
            languages: List of target language codes, e.g. ``["en", "es"]``.
                       ``"pt"`` is always produced (it is the revised original).
            delay_seconds: Sleep time between consecutive API calls.
            progress_callback: Called as ``(current, total, message)`` after
                               each chunk is processed.
            stop_event: ``threading.Event``; processing stops if set.

        Returns:
            Mapping of language code to absolute output file path,
            e.g. ``{"pt": "/path/roteiro_pt.txt", "en": "/path/roteiro_en.txt"}``.

        Raises:
            ValueError: When the API key is invalid or the connection fails.
        """
        # --- 1. Inicializar e validar o cliente Gemini ---
        from google import genai  # type: ignore[import]

        logger.info(f"Inicializando cliente Gemini (modelo: {self.model_name})")
        try:
            self._client = genai.Client(api_key=api_key)
            from google.genai import types  # type: ignore[import]
            
            config = None
            if thinking_level and "gemini-3" in self.model_name:
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
                )

            # Chamada mínima para validar a chave antes de processar o roteiro inteiro
            self._client.models.generate_content(
                model=self.model_name,
                contents="ok",
                config=config
            )
            logger.info("API Key do Gemini validada com sucesso.")
        except Exception as exc:
            raise ValueError(
                f"Não foi possível conectar ao Gemini. Verifique sua API Key.\nDetalhe: {exc}"
            ) from exc

        # --- 2. Parse do arquivo de entrada ---
        txt_path_obj = Path(txt_path)
        logger.info(f"Lendo arquivo de entrada: {txt_path_obj}")
        raw_text = txt_path_obj.read_text(encoding="utf-8-sig")
        paragraphs: list[tuple[int, str]] = []  # (número, texto)
        for line in raw_text.splitlines():
            line = line.strip()
            if not line: continue
            # Mais flexível: aceita "1. Texto", "1.Texto", "1 Texto", "01. Texto"
            m = re.match(r"^(\d+)[.\s-]*\s*(.+)$", line)
            if m:
                paragraphs.append((int(m.group(1)), m.group(2)))
            else:
                # Se não tem número, mas a linha não é vazia, tenta atribuir o próximo número
                last_num = paragraphs[-1][0] if paragraphs else 0
                paragraphs.append((last_num + 1, line))

        if not paragraphs:
            logger.warning("Nenhum parágrafo encontrado no arquivo. Encerrando.")
            return {}

        total_paragraphs = len(paragraphs)
        logger.info(f"Total de parágrafos encontrados: {total_paragraphs}")

        # --- 3. Cache ---
        cache_dir = Path("gemini_cache")
        cache_dir.mkdir(exist_ok=True)
        # Use content explicitly so edits bypass old cache
        cache_key = hashlib.md5(raw_text.encode("utf-8")).hexdigest()[:8]
        cache_file = cache_dir / f"{cache_key}_{txt_path_obj.stem}.json"
        cache = self._load_cache(cache_file, txt_path, total_paragraphs, languages)

        # --- 4. Calcular total de chunks para progress ---
        self._revision_prompt_tmpl = revision_prompt or _REVISION_SYSTEM
        self._translation_prompt_tmpl = translation_prompt or _TRANSLATION_SYSTEM
        self._overlap = overlap
        self._thinking_level = thinking_level
        self._media_resolution = media_resolution
        
        chunks = self._build_chunks(paragraphs, chunk_size=chunk_size)

        n_chunks = len(chunks)
        n_langs = len(languages)
        total_steps = n_chunks + (n_chunks * n_langs)
        current_step = 0

        def _report(message: str):
            if progress_callback:
                progress_callback(current_step, total_steps, message)

        # --- 5. Etapa 1 — Revisão (sempre executa, a menos que já concluída) ---
        if not cache.get("revision_complete", False):
            logger.info("Iniciando etapa de revisão...")
            for i, chunk in enumerate(chunks):
                if stop_event and stop_event.is_set():
                    logger.info("Cancelamento solicitado durante revisão.")
                    self._save_cache(cache_file, cache)
                    break

                current_step += 1
                _report(f"Revisando chunk {i + 1} de {n_chunks}...")

                context_before, current_block, context_after = self._split_chunk_context(
                    paragraphs, chunk
                )
                revised = self._call_with_retry(
                    prompt=self._build_revision_prompt(
                        context_before, current_block, context_after
                    ),
                    expected_count=len(current_block),
                    fallback_texts=[p[1] for p in current_block],
                )
                # Deduplicação: última gravação vence
                for (num, _), new_text in zip(current_block, revised):
                    cache["revised"][str(num)] = new_text

                self._save_cache(cache_file, cache)
                logger.info(f"Revisão — chunk {i + 1}/{n_chunks} concluído.")
                time.sleep(delay_seconds)
            else:
                # Loop completou sem break (sem cancelamento)
                cache["revision_complete"] = True
                self._save_cache(cache_file, cache)
                logger.info("Etapa de revisão concluída.")
        else:
            logger.info("Revisão já concluída (cache encontrado). Pulando.")
            current_step += n_chunks  # avança o progresso

        # Montar lista de parágrafos revisados em ordem
        revised_paragraphs: list[tuple[int, str]] = [
            (num, cache["revised"].get(str(num), text)) for (num, text) in paragraphs
        ]

        # --- 6. Etapa 2 — Tradução por idioma ---
        for lang in languages:
            if stop_event and stop_event.is_set():
                logger.info("Cancelamento solicitado antes de iniciar tradução.")
                break

            lang_name = LANGUAGE_NAMES.get(lang, lang)
            if cache.get("translations_complete", {}).get(lang, False):
                logger.info(f"Tradução {lang.upper()} já concluída (cache). Pulando.")
                current_step += n_chunks
                continue

            logger.info(f"Iniciando tradução para {lang_name}...")
            if lang not in cache["translations"]:
                cache["translations"][lang] = {}

            for i, chunk in enumerate(chunks):
                if stop_event and stop_event.is_set():
                    logger.info(f"Cancelamento durante tradução {lang.upper()}.")
                    self._save_cache(cache_file, cache)
                    break

                current_step += 1
                _report(f"Traduzindo {lang.upper()}: chunk {i + 1} de {n_chunks}...")

                context_before, current_block_orig, context_after = (
                    self._split_chunk_context(paragraphs, chunk)
                )
                
                source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
                target_name = LANGUAGE_NAMES.get(lang, lang)

                # Usar os textos revisados para traduzir
                current_block_revised = [
                    (num, cache["revised"].get(str(num), text))
                    for (num, text) in current_block_orig
                ]
                ctx_before_rev = [
                    cache["revised"].get(str(num), text) for (num, text) in context_before
                ]
                ctx_after_rev = [
                    cache["revised"].get(str(num), text) for (num, text) in context_after
                ]

                translated = self._call_with_retry(
                    prompt=self._build_translation_prompt(
                        target_name, source_name, ctx_before_rev, current_block_revised, ctx_after_rev
                    ),
                    expected_count=len(current_block_revised),
                    fallback_texts=[p[1] for p in current_block_revised],
                )

                for (num, _), new_text in zip(current_block_revised, translated):
                    cache["translations"][lang][str(num)] = new_text

                self._save_cache(cache_file, cache)
                logger.info(f"Tradução {lang.upper()} — chunk {i + 1}/{n_chunks} concluído.")
                time.sleep(delay_seconds)
            else:
                cache.setdefault("translations_complete", {})[lang] = True
                self._save_cache(cache_file, cache)
                logger.info(f"Tradução {lang.upper()} concluída.")

        # --- 7. Salvar arquivos de saída ---
        output_paths: dict[str, str] = {}
        out_dir = txt_path_obj.parent
        stem = txt_path_obj.stem

        # PT (revisado)
        pt_dir = out_dir / "pt"
        pt_dir.mkdir(parents=True, exist_ok=True)
        pt_path = pt_dir / f"{stem}_pt.txt"
        self._write_output(pt_path, revised_paragraphs)
        output_paths["pt"] = str(pt_path)
        logger.info(f"Arquivo PT salvo na pasta: {pt_path}")

        # Idiomas solicitados
        for lang in languages:
            if lang not in cache["translations"]:
                continue
            lang_paras: list[tuple[int, str]] = [
                (num, cache["translations"][lang].get(str(num), text))
                for (num, text) in revised_paragraphs
            ]
            lang_dir = out_dir / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            lang_path = lang_dir / f"{stem}_{lang}.txt"
            self._write_output(lang_path, lang_paras)
            output_paths[lang] = str(lang_path)
            logger.info(f"Arquivo {lang.upper()} salvo na pasta: {lang_path}")

        if progress_callback:
            progress_callback(total_steps, total_steps, "Processamento concluído.")
        return output_paths

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _build_chunks(
        self, paragraphs: list[tuple[int, str]], chunk_size: int = 12
    ) -> list[list[int]]:
        """Divide a lista de parágrafos em blocos de tamanho fixo."""
        nums = [p[0] for p in paragraphs]
        return [nums[i : i + chunk_size] for i in range(0, len(nums), chunk_size)]

    def _split_chunk_context(
        self, paragraphs: list[tuple[int, str]], chunk_nums: list[int]
    ) -> tuple[list[str], list[tuple[int, str]], list[str]]:
        """Dado um chunk, retorna o contexto anterior, o bloco atual e o contexto posterior."""
        overlap = getattr(self, "_overlap", 2)
        start_idx = -1
        for i, (num, _) in enumerate(paragraphs):
            if num == chunk_nums[0]:
                start_idx = i
                break
        
        end_idx = start_idx + len(chunk_nums)
        
        # Contexto anterior (overlap parágrafos antes)
        before_start = max(0, start_idx - overlap)
        context_before = [p[1] for p in paragraphs[before_start:start_idx]]
        
        # Bloco atual
        current_block = paragraphs[start_idx:end_idx]
        
        # Contexto posterior (overlap parágrafos depois)
        after_end = min(len(paragraphs), end_idx + overlap)
        context_after = [p[1] for p in paragraphs[end_idx:after_end]]
        
        return context_before, current_block, context_after

    def _format_block(self, block: list[tuple[int, str]]) -> str:
        return "\n".join(f"{num}. {text}" for num, text in block)

    def _build_revision_prompt(
        self, context_before: list[str], current_block: list[tuple[int, str]], context_after: list[str]
    ) -> str:
        ctx_b = "\n".join(context_before) if context_before else "(início do texto)"
        ctx_a = "\n".join(context_after) if context_after else "(fim do texto)"
        blk = self._format_block(current_block)
        n = len(current_block)
        return self._revision_prompt_tmpl.format(
            context_before=ctx_b,
            context_after=ctx_a,
            current_block_formatted=blk,
            n=n,
        )

    def _build_translation_prompt(
        self,
        language_name: str,
        source_language_name: str,
        context_before: list[str],
        current_block: list[tuple[int, str]],
        context_after: list[str],
    ):
        ctx_b = "\n".join(context_before) if context_before else "(início do texto)"
        ctx_a = "\n".join(context_after) if context_after else "(fim do texto)"
        blk = self._format_block(current_block)
        n = len(current_block)
        return self._translation_prompt_tmpl.format(
            language_name=language_name,
            source_language_name=source_language_name,
            context_before=ctx_b,
            context_after=ctx_a,
            current_block_formatted=blk,
            n=n,
        )

    def _call_with_retry(
        self,
        prompt: str,
        expected_count: int,
        fallback_texts: list[str],
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> list[str]:
        """Call the Gemini API with validation and up to max_retries retries."""
        from google.genai import types  # type: ignore[import]
        
        # Build config
        config_kwargs = {}
        if "gemini-3" in self.model_name:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self._thinking_level
            )
            # media_resolution can also be set globally in generation_config
            # but ultra_high might not be available globally.
            # For text-only tasks, this is mostly ignored but we include it if set.
            if self._media_resolution:
                # Some versions might use different structures, following the provided doc:
                # media_resolution is usually per-part, but doc mentions global option.
                pass 

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
                )

                raw = response.text.strip()
                # Remover blocos markdown se o modelo os incluir inadvertidamente
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
                    raw = re.sub(r"\n?```$", "", raw)

                data = json.loads(raw)
                items = data.get("paragrafos", data.get("paragraphs", []))

                if not isinstance(items, list):
                    raise ValueError("Resposta 'paragrafos' não é uma lista.")
                if len(items) != expected_count:
                    raise ValueError(
                        f"Esperado {expected_count} itens, recebido {len(items)}."
                    )
                # Validar que nenhum item é vazio ou None
                for idx, item in enumerate(items):
                    if not item or not isinstance(item, str):
                        raise ValueError(f"Item {idx} inválido: {item!r}")

                return items

            except Exception as exc:
                is_429 = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                
                # Exponential backoff for 429
                wait = retry_delay
                if is_429:
                    wait = retry_delay * (2 ** (attempt - 1))
                    logger.warning(f"Quota Gemini atingida (429). Aguardando {wait}s p/ tentativa {attempt+1}...")
                
                logger.warning(
                    f"Tentativa {attempt}/{max_retries} falhou: {exc}. "
                    + ("Tentando novamente..." if attempt < max_retries else "Usando texto original.")
                )
                
                if attempt < max_retries:
                    time.sleep(wait)


        # Fallback: retornar textos originais sem modificação
        logger.error(
            f"Esgotadas {max_retries} tentativas. Usando {expected_count} parágrafos originais."
        )
        return fallback_texts

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(
        self,
        cache_file: Path,
        txt_path: str,
        total_paragraphs: int,
        languages: list[str],
    ) -> dict:
        """Load existing cache or return a fresh structure."""
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                # Garantir que as traduções dos novos idiomas existem no cache
                for lang in languages:
                    data.setdefault("translations", {}).setdefault(lang, {})
                    data.setdefault("translations_complete", {})[lang] = (
                        data.get("translations_complete", {}).get(lang, False)
                    )
                logger.info(f"Cache carregado: {cache_file}")
                return data
            except Exception as exc:
                logger.warning(f"Não foi possível carregar cache ({exc}). Iniciando do zero.")

        return {
            "source_file": txt_path,
            "total_paragraphs": total_paragraphs,
            "revised": {},
            "translations": {lang: {} for lang in languages},
            "revision_complete": False,
            "translations_complete": {lang: False for lang in languages},
        }

    def _save_cache(self, cache_file: Path, cache: dict) -> None:
        """Persist the current cache state to disk."""
        try:
            cache_file.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error(f"Falha ao salvar cache: {exc}")

    def _write_output(self, path: Path, paragraphs: list[tuple[int, str]]) -> None:
        """Write numbered paragraphs to the output file."""
        lines = [f"{num}. {text}" for num, text in sorted(paragraphs, key=lambda x: x[0])]
        path.write_text("\n".join(lines), encoding="utf-8")
