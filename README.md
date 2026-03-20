# 🎬 Manhwa Video Creator

O **Manhwa Video Creator** é uma ferramenta profissional de automação para criação de vídeos estilo "Manhwa Recap" ou "Narrativa Visual". Ele combina tecnologias de ponta em **TTS (Text-to-Speech)**, **NLP (Processamento de Linguagem Natural)** e **Composição de Vídeo** para transformar roteiros simples em vídeos dinâmicos com narração humana e efeitos visuais automáticos.

---

## ✨ Funcionalidades Principais

*   **🎙️ Motores de Voz Avançados:**
    *   **Chatterbox (V2):** Vozes expressivas com suporte a clonagem (Zero-shot) e modelos Turbo/Multilingual.
    *   **Kokoro (Local):** Motor de voz ultra-rápido e leve com alta fidelidade.
*   **🧩 Inteligência de Texto (SpaCy):** Processamento automático para adicionar pausas naturais e melhorar a fluência da narração.
*   **🖼️ Efeito Ken Burns Automático:** Aplica zooms e movimentos de câmera (Pan) suaves para dar vida a imagens estáticas.
*   **✅ Verificação de Qualidade (Whisper):** Transcreve o áudio gerado em tempo real e compara com o texto original, refazendo a narração automaticamente se houver erros de pronúncia.
*   **🚀 Otimização Blackwell (RTX 5070 Ti/40 Series):** Totalmente otimizado para GPUs NVIDIA modernas usando `bfloat16`, `TF32` e limpeza estratégica de VRAM para sessões longas.
*   **🛠️ Pipeline Tudo-em-Um:** Gera o áudio, aplica trilha sonora de fundo e renderiza o vídeo final usando aceleração de hardware (NVENC).

---

## 🚀 Como Instalar

### 1. Pré-requisitos
*   **Python 3.10 ou superior.**
*   **FFmpeg** instalado no seu sistema e adicionado ao seu PATH.
*   **NVIDIA GPU** com pelo menos 8GB de VRAM (Recomendado para melhor performance).

### 2. Instalação Automática (Windows)
Basta baixar o projeto e clicar duas vezes no arquivo:
```bash
start.bat
```
Este script irá criar um ambiente virtual, instalar todas as dependências (incluindo PyTorch com suporte a CUDA) e baixar os modelos de linguagem necessários.

### 3. Instalação Manual
Se preferir fazer manualmente:
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/manhwa-video-creator.git
cd manhwa-video-creator

# Crie e ative um ambiente virtual
python -m venv venv
venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Baixe o modelo do SpaCy (Português)
python -m spacy download pt_core_news_lg
```

---

## 🛠️ Como Usar

1.  **Aba Áudio:** Adicione seus arquivos `.txt` (separe parágrafos com uma linha em branco).
2.  **Aba TTS:** Escolha o motor (Chatterbox ou Kokoro) e a voz desejada. O modelo será carregado automaticamente no fundo.
3.  **Aba Imagens:** Adicione as imagens que ilustram cada parágrafo.
4.  **Aba Vídeo:** Configure os efeitos (Zoom, Pan, Transições) e clique em **🎬 Gerar Vídeo**.

---

## 💻 Tech Stack

*   **Interface:** PySide6 (Qt para Python)
*   **TTS:** Chatterbox-V2, Kokoro-TTS
*   **Vídeo:** FFmpeg (via Subprocess & FFmpeg-python), Pillow (Composição Ken Burns)
*   **Deep Learning:** PyTorch (CUDA 12.4)
*   **STT (Verification):** OpenAI Whisper
*   **NLP:** SpaCy

---

## 📄 Créditos
Desenvolvido para criadores de conteúdo que buscam velocidade e qualidade profissional na produção de vídeos de manhwas, webtoons e audiobooks.
