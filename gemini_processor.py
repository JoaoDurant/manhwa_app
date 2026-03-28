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
_REVISION_SYSTEM = """Você é um editor literário especialista em transformar roteiros brutos de manhwa e webtoon em narrações envolventes e profissionais.

Sua tarefa: reescrever e refinar o BLOCO ATUAL de parágrafos para que soem como um audiobook de alta qualidade ou uma narração de vídeo profissional.

REGRAS DE OURO:
1. **TRANSFORMAÇÃO CRIATIVA**: Não se limite a corrigir gramática. Melhore o vocabulário, crie suspense e use uma linguagem que prenda o ouvinte. Pode expandir ligeiramente o texto para melhorar a imersão.
2. **FLUIDEZ NARRATIVA**: Conecte os parágrafos. Se um parágrafo termina com uma pergunta ou um cliffhanger, garanta que o próximo mantenha o ritmo.
3. **LIMPEZA TOTAL**: Remova TODA a "sujeira" visual: símbolos (*, #, →, — ou [ ]), nomes de personagens antes da fala (Ex: "João: Olá" -> "Olá"), onomatopeias gráficas e descrições técnicas de cena.
4. **NATURALIDADE**: Números devem ser escritos por extenso ("7" -> "sete"). Siglas devem ser expandidas ("km" -> "quilômetros").
5. **RITMO**: Quebre frases longas que seriam difíceis de narrar sem pausa para respirar.
6. **ESTRUTURA**: Mantenha exatamente o mesmo número de parágrafos no JSON de saída para sincronia com o sistema.

CONTEXTO DE CONTINUIDADE (Apenas para referência, não edite):
[ANTERIOR]: {context_before}
[PRÓXIMO]: {context_after}

---
BLOCO ATUAL PARA REESCREVER:
{current_block_formatted}

Responda APENAS com JSON válido:
{{"paragrafos": ["texto revisado 1", "texto revisado 2", ...]}}

Gere exatamente {n} elementos."""

_TRANSLATION_SYSTEM = """Você é um tradutor literário de elite, especializado em localizar roteiros de manhwa para narração em áudio no idioma {language_name}.

Sua missão: traduzir o conteúdo revisado mantendo a emoção, a gíria e o "feeling" do original, mas adaptando para a cultura do idioma alvo.

REGRAS DE OURO:
1. **TRADUÇÃO EMOÇÃO-A-EMOÇÃO**: Não traduza literalmente. Use expressões naturais do idioma alvo ({language_name}) que transmitam o mesmo impacto emocional.
2. **NATURALIDADE DE ÁUDIO**: O texto traduzido deve ser fácil de ler e soar como uma conversa ou narração natural, não como um texto traduzido pelo Google.
3. **PRESERVAÇÃO DE NOMES**: Mantenha nomes próprios de personagens e lugares como estão, a menos que haja uma tradução oficial consagrada.
4. **COESÃO**: Mantenha a ligação entre parágrafos (cohesion) para que o ouvinte não sinta pulos entre as faixas de áudio.
5. **ESTRUTURA**: Mantenha exatamente o mesmo número de parágrafos.

CONTEXTO DE CONTINUIDADE (Referência do original):
[ANTERIOR]: {context_before}
[PRÓXIMO]: {context_after}

---
BLOCO ATUAL PARA TRADUZIR:
{current_block_formatted}

Responda APENAS com JSON válido:
{{"paragrafos": ["texto traduzido 1", "texto traduzido 2", ...]}}

Gere exatamente {n} elementos."""


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
        self._revision_prompt_tmpl = _REVISION_SYSTEM
        self._translation_prompt_tmpl = _TRANSLATION_SYSTEM
        self._overlap = 2
        self._thinking_level = "high"
        self._media_resolution = "media_resolution_high"

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
        revision_prompt: Optional[str] = None,
        translation_prompt: Optional[str] = None,
        chunk_size: int = 12,
        overlap: int = 2,
        thinking_level: str = "high",
        media_resolution: str = "media_resolution_high",
        per_language_prompts: Optional[dict[str, str]] = None,
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
        # --- 1. Normalizar e validar o pool de chaves ---
        if isinstance(api_key, str):
            api_pool = [{"alias": "Principal", "key": api_key}] if api_key.strip() else []
        elif isinstance(api_key, dict):
            # Compatibilidade com formato antigo {"rev": "...", "en": "...", ...}
            api_pool = [{"alias": k, "key": v} for k, v in api_key.items() if v and v.strip()]
        elif isinstance(api_key, list):
            api_pool = [e for e in api_key if isinstance(e, dict) and e.get("key", "").strip()]
        else:
            api_pool = []

        if not api_pool:
            raise ValueError("Nenhuma API Key válida foi fornecida.")

        from google import genai  # type: ignore[import]

        n_keys = len(api_pool)
        logger.info(f"Inicializando Gemini (modelo: {self.model_name}) com {n_keys} chave(s): {[e['alias'] for e in api_pool]}")
        try:
            from google.genai import types  # type: ignore[import]
            
            is_thinking_model = any(k in self.model_name.lower() for k in ["thinking", "gemini-3.1", "gemini-3-pro"])
            config_kwargs = {}
            if thinking_level and is_thinking_model:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
            config_kwargs["response_mime_type"] = "application/json"
            config = types.GenerateContentConfig(**config_kwargs)

            # Valida apenas a primeira chave
            test_client = genai.Client(api_key=api_pool[0]["key"])
            test_client.models.generate_content(
                model=self.model_name,
                contents='Generate a single JSON word: {"test": "ok"}',
                config=config
            )
            logger.info(f"API Key '{api_pool[0]['alias']}' validada com sucesso.")
        except Exception as exc:
            raise ValueError(
                f"Não foi possível conectar ao Gemini com a chave '{api_pool[0]['alias']}'.\nDetalhe: {exc}"
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
        # Use content + prompts + params explicitly so edits bypass old cache
        base_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()[:8]
        
        # Include all settings that affect output
        settings_payload = {
            "rev_prompt": revision_prompt,
            "trans_prompt": translation_prompt,
            "per_lang": per_language_prompts,
            "chunk": chunk_size,
            "overlap": overlap,
            "model": self.model_name
        }
        import json
        settings_str = json.dumps(settings_payload, sort_keys=True)
        settings_hash = hashlib.md5(settings_str.encode("utf-8")).hexdigest()[:8]

        cache_key = f"{base_hash}_{settings_hash}"
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
                    api_pool=api_pool,
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

                # Use the custom prompt for this language if provided, else generic
                current_translation_prompt = (per_language_prompts or {}).get(lang) or self._translation_prompt_tmpl

                translated = self._call_with_retry(
                    api_pool=api_pool,
                    prompt=self._build_translation_prompt(
                        target_name, source_name, ctx_before_rev, current_block_revised, ctx_after_rev,
                        prompt_override=current_translation_prompt
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
    ) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, str]]]:
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
        context_before = paragraphs[before_start:start_idx]
        
        # Bloco atual
        current_block = paragraphs[start_idx:end_idx]
        
        # Contexto posterior (overlap parágrafos depois)
        after_end = min(len(paragraphs), end_idx + overlap)
        context_after = paragraphs[end_idx:after_end]
        
        return context_before, current_block, context_after

    def _format_block(self, block: list[tuple[int, str]]) -> str:
        return "\n".join(f"{num}. {text}" for num, text in block)

    def _build_revision_prompt(
        self, context_before: list[tuple[int, str]], current_block: list[tuple[int, str]], context_after: list[tuple[int, str]]
    ) -> str:
        ctx_b = "\n".join(t for n, t in context_before) if context_before else "(início do texto)"
        ctx_a = "\n".join(t for n, t in context_after) if context_after else "(fim do texto)"
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
        context_before: list[tuple[int, str]],
        current_block: list[tuple[int, str]],
        context_after: list[tuple[int, str]],
        prompt_override: str = None,
    ):
        ctx_b = "\n".join(p[1] for p in context_before) if context_before else "(início do texto)"
        ctx_a = "\n".join(p[1] for p in context_after) if context_after else "(fim do texto)"
        blk = self._format_block(current_block)
        n = len(current_block)
        
        tmpl = prompt_override or self._translation_prompt_tmpl
        return tmpl.format(
            language_name=language_name,
            source_language_name=source_language_name,
            context_before=ctx_b,
            context_after=ctx_a,
            current_block_formatted=blk,
            n=n,
        )

    def normalize_gemini_response(self, response) -> str:
        """
        Garante que a resposta seja validada e convertida para string
        independentemente do formato da engine SDK ou de falhas json.
        """
        if isinstance(response, str):
            return response.strip()
            
        if hasattr(response, "text") and hasattr(response, "candidates"):
            try:
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text.strip()
            except Exception as e:
                logger.debug(f"[Normalize] Falha parts: {e}")
            try:
                if response.text:
                    return response.text.strip()
            except Exception as e:
                logger.debug(f"[Normalize] Falha text: {e}")
                
        if isinstance(response, dict):
            if "candidates" in response and response["candidates"]:
                try:
                    return response["candidates"][0]["content"]["parts"][0]["text"].strip()
                except Exception:
                    pass
            return json.dumps(response, ensure_ascii=False)

        if isinstance(response, list):
            if len(response) > 0:
                first = response[0]
                if isinstance(first, str):
                    return first.strip()
                elif isinstance(first, dict):
                    return json.dumps(first, ensure_ascii=False)
            return ""

        raise TypeError(f"Formato não suportado para normalização: {type(response)}")

    def validate_output(self, text: str, expected_lines: int) -> list[str]:
        """
        Valida que a saída tem a quantidade esperada de parágrafos.
        Substitui de forma inteligente se o JSON estiver quebrado (fallback raw text).
        """
        import re
        lines = []
        for m in re.finditer(r"^\s*\d+[.\-]?\s+(.+)$", text, flags=re.MULTILINE):
            lines.append(m.group(1).strip())
            
        if len(lines) == expected_lines:
            return lines
            
        # Fallback agressivo de plain-text puro
        for separator in ['\n\n', '\n']:
            parts = [p.strip() for p in text.split(separator) if p.strip()]
            if len(parts) == expected_lines:
                return parts
            
        raise ValueError(f"Esperado {expected_lines} itens válidos numéricos, mas extraídos {len(lines)}.")

    def _call_with_retry(
        self,
        api_pool: list,   # list[{"alias": str, "key": str}]
        prompt: str,
        expected_count: int,
        fallback_texts: list[str],
        max_retries: int = 6,
        retry_delay: float = 5.0,
    ) -> list[str]:
        """Chama a API Gemini com rotação sequencial de chaves em caso de 429."""
        import google.genai as genai
        from google.genai import types  # type: ignore[import]

        config_kwargs = {"response_mime_type": "application/json"}
        is_thinking_model = "thinking" in self.model_name.lower() or "gemini-2.0-flash-thinking" in self.model_name.lower()
        if is_thinking_model:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self._thinking_level)

        # Índice rotativo — começa sempre na chave 0 (Principal)
        pool_idx = 0
        normal_attempts = 0  # contagem de tentativas que NÃO são troca de chave

        while normal_attempts < max_retries:
            entry = api_pool[pool_idx]
            current_alias = entry["alias"]
            current_key   = entry["key"]
            client = genai.Client(api_key=current_key)

            try:
                formatted_contents = [{"parts": [{"text": prompt}]}]
                try:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=formatted_contents,
                        config=types.GenerateContentConfig(**config_kwargs)
                    )
                except Exception as api_exc:
                    err_str = str(api_exc).upper()
                    if "400" in err_str or "INVALID_ARGUMENT" in err_str or "API_KEY_INVALID" in err_str:
                        logger.error(f"Erro Crítico API 400 na conta '{current_alias}': {api_exc}")
                        raise StopIteration(f"Erro 400 em '{current_alias}'. Sem retry.") from api_exc
                    raise

                try:
                    raw = self.normalize_gemini_response(response)
                except Exception as norm_exc:
                    logger.debug(f"Falha na normalização: {norm_exc}")
                    raw = ""

                extracted = self._extract_json(raw)
                data = None
                items = None
                if extracted:
                    try:
                        data = json.loads(extracted)
                        if isinstance(data, list):
                            if data and all(isinstance(x, str) for x in data):
                                items = data
                                data = {"paragrafos": items}
                            elif data:
                                data = data[0]
                            else:
                                raise ValueError("Array JSON vazia retornada do Gemini.")
                        if not isinstance(data, dict):
                            raise TypeError(f"Dado JSON incorreto: {type(data)}")
                        if not items:
                            items = data.get("paragrafos", data.get("paragraphs", []))
                            if not items and len(data.values()) == expected_count:
                                items = [str(v) for v in data.values()]
                    except Exception as json_err:
                        logger.debug(f"Falha decodificando payload JSON: {json_err}")

                if not items or len(items) != expected_count:
                    try:
                        items = self.validate_output(raw, expected_count)
                    except Exception as val_exc:
                        raise ValueError(f"Parsing JSON e Expressão de Parágrafos falharam: {val_exc}")

                import re
                return [re.sub(r'^\s*\d+[\.\-\)\]:]?\s+', '', str(item)).strip() for item in items]

            except StopIteration as stop_err:
                logger.error(str(stop_err))
                break

            except Exception as exc:
                err_str = str(exc)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()

                if is_429 and len(api_pool) > 1:
                    next_idx = (pool_idx + 1) % len(api_pool)
                    if next_idx != pool_idx:  # ainda há outra chave
                        next_alias = api_pool[next_idx]["alias"]
                        logger.warning(
                            f"Cota da conta '{current_alias}' esgotada (429). "
                            f"Trocando imediatamente para '{next_alias}' "
                            f"({next_idx + 1}/{len(api_pool)})..."
                        )
                        pool_idx = next_idx
                        continue  # sem sleep — troca instantânea!

                # Sem fallback ou já rodou todas as chaves
                normal_attempts += 1
                wait = retry_delay
                if is_429:
                    wait = retry_delay * (2 ** (normal_attempts - 1))
                    logger.warning(f"Todas as contas atingiram o limite. Aguardando {wait:.0f}s...")
                logger.warning(
                    f"Tentativa {normal_attempts}/{max_retries} falhou em '{current_alias}': {err_str[:200]}. "
                    + ("(Aguardando retry...)" if normal_attempts < max_retries else "(Desistindo)")
                )
                if normal_attempts < max_retries:
                    time.sleep(wait)

        logger.error(f"Esgotadas todas as tentativas. Retornando parágrafos originais.")
        return fallback_texts

    def _extract_json(self, text: str) -> Optional[str]:
        """Extrai o conteúdo entre o primeiro { e o último } na string."""
        try:
            # Especial: Remova tudo antes do primeiro { ou [ e depois do último } ou ]
            # Usando regex que captura o maior bloco estruturado possível
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                return match.group(0).strip()
            return None
        except Exception:
            return None

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
