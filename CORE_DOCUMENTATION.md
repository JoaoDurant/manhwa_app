# 📜 Documentação Técnica: Manhwa Video Creator (v6)

Esta documentação descreve a arquitetura do sistema, o papel de cada arquivo e como o fluxo de dados opera para transformar scripts de texto em vídeos cinematográficos de manhwa, otimizados para a arquitetura **GeForce RTX 5070 Ti** e CPUs Intel de 14ª Geração.

---

## 🏗️ Arquitetura do Sistema

O projeto segue uma estrutura modular onde a interface gráfica (Frontend) delega tarefas pesadas para pipelines especializados (Orquestradores), que por sua vez utilizam drivers de hardware de baixo nível (Engines).

### 1. Inicialização e Ambiente
| Arquivo | Descrição |
| :--- | :--- |
| `run_manhwa_app.py` | **O Launcher.** Verifica o ambiente virtual (venv), detecta a placa de vídeo, confere se o PyTorch é compatível com RTX 50 e reinicia o app no ambiente correto se necessário. |
| `config.py` | Gerencia o carregamento e salvamento de configurações (`config.yaml`), cuidando de caminhos de diretório e parâmetros globais. |
| `utils.py` | Gigantesco canivete suíço. Contém funções de processamento de áudio (trim de silêncio, normalização), manipulação de nomes de arquivos e suporte a FFmpeg. |

### 2. O Motor de Áudio (Core)
| Arquivo | Descrição |
| :--- | :--- |
| `engine.py` | **O Cérebro TTS.** Unifica o Chatterbox (Turbo/Multilingual) e o Kokoro. Implementa as otimizações de `max-autotune` (Triton) e gestão dinâmica de VRAM. |
| `kokoro_utils.py` | Utilitários específicos para o motor Kokoro, como limpeza de fonemas e mapeamento de idiomas. |

### 3. A Interface Gráfica (manhwa_app/)
| Arquivo | Descrição |
| :--- | :--- |
| `manhwa_app/app.py` | **O Coração da UI.** Quase 200KB de código PySide6 (Qt). Gerencia as abas, a visualização de imagens, o log em tempo real e a inicialização de threads de carregamento de modelo. |
| `_patch_queue.py` | Script utilitário que "injeta" funcionalidades de fila no `app.py` sem precisar reescrever o arquivo inteiro manualmente. |

### 4. Pipelines de Processamento
| Arquivo | Descrição |
| :--- | :--- |
| `manhwa_app/audio_pipeline.py` | **Orquestrador de Áudio.** Pega um arquivo .txt, divide em parágrafos, envia para o `engine.py`, valida a qualidade via Whisper e aplica efeitos de pós-processamento de voz. |
| `manhwa_app/video_pipeline.py` | **Orquestrador de Vídeo.** O motor que fala com o FFmpeg. Cria o efeito **Ken Burns** (zoom/pan suave), gera o fundo borrado dinâmico e mixa a trilha sonora com **Auto-Ducking**. |
| `manhwa_app/audio_fx.py` | Aplica filtros de áudio (reverb, EQ, compressão) para dar um ar mais "cinematográfico" à narração. |
| `manhwa_app/advanced_text_processor.py` | Melhora a fluência do texto antes de enviar para o TTS, expandindo abreviações e corrigindo pontuação. |

---

## 🌊 Fluxo de Trabalho (Data Pipeline)

1.  **Entrada**: O usuário carrega um script `.txt` e uma pasta de imagens.
2.  **Processamento de Texto**: O `audio_pipeline` limpa o texto via `advanced_text_processor`.
3.  **Sintese de Áudio**: O `engine.py` gera áudio em pedaços (parágrafos). A RTX 5070 Ti usa `max-autotune` para acelerar este passo.
4.  **Validação**: O `Whisper` (opcional) transcreve o áudio para garantir que nada foi pulado.
5.  **Composição de Vídeo**: O `video_pipeline` cria clips `.mp4` individuais para cada parágrafo com efeitos de câmera e depois os une.
6.  **Mixagem Final**: A música de fundo é adicionada com volume reduzido automaticamente durante as falas.

---

## 🔥 Otimizações RTX 5070 Ti (Blackwell)

Para aproveitar o máximo do seu hardware, implementamos:
- **Triton Torch Compile (Max-Autotune)**: Reduz o tempo de geração do Chatterbox significativamente após os primeiros segundos.
- **NVENC P4**: O pipeline de vídeo usa codificação por hardware NVIDIA de 4ª/5ª geração para renderizar vídeos 60fps quase instantaneamente.
- **Pinned Memory Storage**: Transferência ultra-rápida de frames entre RAM e VRAM para o efeito Ken Burns.
- **DType Auto-Selection**: Uso de `bfloat16` onde suportado para dobro de performance com mesma memória.
