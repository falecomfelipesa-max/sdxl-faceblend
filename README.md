# SDXL + IP-Adapter FaceID — Blend de Identidade Dual

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/falecomfelipesa-max/sdxl-faceblend/blob/main/FaceBlend_Colab.ipynb)
[![Lint](https://github.com/falecomfelipesa-max/sdxl-faceblend/actions/workflows/ci.yml/badge.svg)](https://github.com/falecomfelipesa-max/sdxl-faceblend/actions)

> **Rodando na nuvem? Clique no badge acima.** Não precisa de GPU local.

---

## ☁️ Rodando na Nuvem (Google Colab — sem GPU local)

Esta é a forma recomendada. Compatível com **Google One AI Premium / Ultra**
(Colab Pro+ com GPU A100 80 GB).

| Passo | O que fazer |
|-------|-------------|
| 1 | Clique no badge **Open in Colab** acima |
| 2 | `Runtime → Change runtime type → A100 GPU` |
| 3 | Execute as células 1 a 4 uma vez (~5 min) |
| 4 | Faça upload das fotos em `Drive/faceblend/faces/` |
| 5 | Edite os caminhos e prompts na **célula 5** e execute |
| 6 | As imagens aparecem inline e são salvas no seu Drive automaticamente |

---

Gera imagens de alta qualidade (1024 × 1024) a partir de prompts de texto,
preservando a identidade de **duas pessoas simultaneamente** via média de
embeddings ArcFace injetados no pipeline Stable Diffusion XL.

---

## Arquitetura

```
Foto A  ──┐
           ├── InsightFace (ArcFace) ──► Média dos embeddings ──► IP-Adapter FaceID
Foto B  ──┘                                                           │
                                                                      ▼
Prompt de texto ──────────────────────────────────────────► SDXL Base ──► Imagem 1024²
```

---

## Pré-requisitos

| Requisito | Versão mínima |
|-----------|---------------|
| Python    | 3.10          |
| CUDA      | 11.8 ou 12.1  |
| VRAM GPU  | ≥ 12 GB (recomendado 16 GB+) |

---

## Instalação

### 1. Ambiente virtual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. PyTorch com CUDA (instale ANTES das outras dependências)

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Demais dependências

```bash
pip install -r requirements.txt
```

---

## Download dos pesos do IP-Adapter FaceID SDXL

```bash
# Opção A: huggingface-cli
huggingface-cli download h94/IP-Adapter-FaceID \
    ip_adapter_faceid_sdxl.bin \
    --local-dir .

# Opção B: wget / curl
wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip_adapter_faceid_sdxl.bin
```

> O arquivo deve ficar em `./ip_adapter_faceid_sdxl.bin` (caminho padrão do script).

---

## Uso

### Estrutura de pastas esperada

```
galactic-sagan/
├── generate_faceblend.py
├── requirements.txt
├── ip_adapter_faceid_sdxl.bin   ← baixar conforme instruções acima
├── faces/
│   ├── pessoa_a.jpg
│   └── pessoa_b.jpg
└── output/                      ← criado automaticamente
```

### Exemplo básico

```bash
python generate_faceblend.py \
  --face-a faces/pessoa_a.jpg \
  --face-b faces/pessoa_b.jpg \
  --prompts \
    "a person in a futuristic space station, cinematic lighting, 8k" \
    "a person in a Renaissance painting style, oil on canvas" \
    "a portrait of a person on the surface of Mars, epic sci-fi"
```

### Exemplo com parâmetros completos

```bash
python generate_faceblend.py \
  --face-a  faces/pessoa_a.jpg \
  --face-b  faces/pessoa_b.jpg \
  --prompts \
    "a professional headshot, studio lighting, sharp focus" \
    "a person hiking in Patagonia, golden hour, wide angle" \
  --output  ./output \
  --steps   40 \
  --cfg     7.5 \
  --seed    1234 \
  --scale   0.8 \
  --device  cuda
```

### Parâmetros CLI

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--face-a` | — | Caminho da 1ª foto (obrigatório) |
| `--face-b` | — | Caminho da 2ª foto (obrigatório) |
| `--prompts` | — | Um ou mais prompts (obrigatório) |
| `--output` | `./output` | Pasta de saída |
| `--steps` | `40` | Inference steps |
| `--cfg` | `7.5` | CFG / Guidance scale |
| `--seed` | `42` | Semente inicial (incrementa por imagem) |
| `--scale` | `0.8` | Força do IP-Adapter (0.0–1.0) |
| `--device` | `cuda` | `cuda` ou `cpu` |
| `--base-model` | `stabilityai/sdxl-base-1.0` | Modelo SDXL base |
| `--vae-model` | `madebyollin/sdxl-vae-fp16-fix` | VAE fp16 |
| `--ip-ckpt` | `./ip_adapter_faceid_sdxl.bin` | Checkpoint IP-Adapter |

---

## Como funciona

### Módulo 1 — Pipeline (`build_pipeline`)
- Carrega o **VAE fp16-fix** para evitar artefatos de cor.
- Inicializa o **SDXL base** em `float16` na GPU.
- Configura o **DDIMScheduler** (boa convergência com 40 steps).
- Ativa `xformers` para reduzir o consumo de VRAM.
- Envolve tudo com **IPAdapterFaceIDXL**.

### Módulo 2 — Embeddings (`average_face_embeddings`)
- Usa **InsightFace buffalo_l** (detector + ArcFace 512-d).
- Extrai o vetor de 512 dimensões de cada rosto.
- Calcula a **média aritmética exata** dos dois vetores.
- Renormaliza para manter a norma unitária.
- Empacota no formato `[1, 1, 512]` esperado pelo IP-Adapter.

### Módulo 3 — Loop de geração (`run_generation_loop`)
- Itera sobre a lista de prompts.
- Para cada prompt: 40 steps · CFG 7.5 · 1024 × 1024.
- Semente incrementada por prompt → resultados reproduzíveis.
- Salva cada imagem como `output_000.png`, `output_001.png`, …

---

## Dicas

- **`--scale 0.8`** é o ponto de equilíbrio entre fidelidade de identidade
  e criatividade do prompt. Valores maiores (ex.: 1.0) reforçam a face;
  valores menores (ex.: 0.5) dão mais liberdade ao texto.
- Use fotos com **rosto frontal bem iluminado** para melhores embeddings.
- Se a VRAM for limitada, adicione `pipe.enable_model_cpu_offload()` em
  `build_pipeline` (reduz velocidade, mas economiza VRAM).

---

## Licença

Distribuído sob os termos das licenças dos modelos utilizados:
[SDXL Community License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) ·
[IP-Adapter License](https://github.com/tencent-ailab/IP-Adapter/blob/main/LICENSE) ·
[InsightFace License](https://github.com/deepinsight/insightface/blob/master/LICENSE)
