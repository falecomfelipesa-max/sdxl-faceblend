"""
generate_faceblend.py
─────────────────────────────────────────────────────────────────────────────
Pipeline SDXL + IP-Adapter FaceID
Fluxo:
  1. Carrega o pipeline SDXL + IP-Adapter FaceID na GPU (CUDA).
  2. Lê DUAS fotos de rosto, extrai embeddings via InsightFace e calcula
     a MÉDIA MATEMÁTICA dos vetores — garantindo identidade mista precisa.
  3. Itera sobre uma lista de prompts, gerando imagens com:
       • 40 inference steps
       • CFG Scale 7.5
       • Resolução 1024 × 1024 px
  4. Salva os resultados em `./output/`.

Requisitos:
  pip install -r requirements.txt
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

# ─── InsightFace ──────────────────────────────────────────────────────────────
import insightface
from insightface.app import FaceAnalysis

# ─── Diffusers / IP-Adapter FaceID ────────────────────────────────────────────
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

# ─── Configurações padrão ────────────────────────────────────────────────────
DEFAULT_BASE_MODEL   = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_VAE_MODEL    = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_IP_CKPT      = "./ip_adapter_faceid_sdxl.bin"  # Baixar manualmente
DEFAULT_OUTPUT_DIR   = "./output"
DEFAULT_STEPS        = 40
DEFAULT_CFG          = 7.5
DEFAULT_WIDTH        = 1024
DEFAULT_HEIGHT       = 1024
DEVICE               = "cuda"          # alterar para "cpu" se não tiver GPU


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 1 — Inicialização do pipeline SDXL + IP-Adapter FaceID
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    base_model: str  = DEFAULT_BASE_MODEL,
    vae_model: str   = DEFAULT_VAE_MODEL,
    ip_ckpt: str     = DEFAULT_IP_CKPT,
    device: str      = DEVICE,
) -> IPAdapterFaceIDXL:
    """
    Constrói e retorna o pipeline SDXL com IP-Adapter FaceID carregado
    inteiramente na GPU (float16 para máxima performance em VRAM).

    Parâmetros
    ----------
    base_model : str
        Hub ID ou caminho local do modelo SDXL base.
    vae_model  : str
        Hub ID ou caminho local do VAE fp16-fix (evita artefatos).
    ip_ckpt    : str
        Caminho para o checkpoint `.bin` do IP-Adapter FaceID SDXL.
    device     : str
        "cuda" (recomendado) ou "cpu".

    Retorna
    -------
    IPAdapterFaceIDXL
        Pipeline pronto para geração.
    """
    print("[1/3] Carregando VAE fp16-fix …")
    vae = AutoencoderKL.from_pretrained(
        vae_model,
        torch_dtype=torch.float16,
    )

    print("[2/3] Carregando pipeline SDXL …")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Scheduler DDIM — melhor convergência com poucos steps
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Otimizações de memória (mantém velocidade em GPUs com < 24 GB VRAM)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    print("[3/3] Carregando IP-Adapter FaceID SDXL …")
    ip_model = IPAdapterFaceIDXL(
        pipe,
        ip_ckpt,
        device=device,
        num_tokens=4,
    )

    print("✓ Pipeline pronto.\n")
    return ip_model


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 2 — Extração de embeddings faciais e cálculo da média
# ══════════════════════════════════════════════════════════════════════════════

def build_face_analyzer(device: str = DEVICE) -> FaceAnalysis:
    """
    Inicializa o analisador InsightFace (buffalo_l) com detecção de rosto
    e extração de embedding (ArcFace 512-d).
    """
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    app = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    # det_size deve ser múltiplo de 32; 640×640 cobre bem fotos de retrato.
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return app


def extract_face_embedding(
    analyzer: FaceAnalysis,
    image_path: str,
) -> np.ndarray:
    """
    Lê a imagem em `image_path`, detecta o rosto de maior score e retorna
    o vetor de embedding ArcFace normalizado (shape: [512]).

    Levanta ValueError se nenhum rosto for detectado.
    """
    # InsightFace espera BGR (formato OpenCV)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    faces = analyzer.get(img_bgr)
    if not faces:
        raise ValueError(f"Nenhum rosto detectado em: {image_path}")

    # Seleciona o rosto com maior área / confiança (geralmente o principal)
    best_face = max(faces, key=lambda f: f.det_score)
    embedding = best_face.normed_embedding          # ndarray float32 [512]
    return embedding.astype(np.float32)


def average_face_embeddings(
    analyzer: FaceAnalysis,
    path_face_a: str,
    path_face_b: str,
) -> torch.Tensor:
    """
    Extrai embeddings das duas fotos e retorna a MÉDIA MATEMÁTICA
    como tensor PyTorch pronto para o IP-Adapter.

    Retorna
    -------
    torch.Tensor  — shape [1, 1, 512]
    """
    print(f"[Face A] Processando: {path_face_a}")
    emb_a = extract_face_embedding(analyzer, path_face_a)

    print(f"[Face B] Processando: {path_face_b}")
    emb_b = extract_face_embedding(analyzer, path_face_b)

    # Média aritmética exata dos dois vetores de 512 dimensões
    emb_avg = (emb_a + emb_b) / 2.0

    # Renormaliza para manter a norma unitária (boa prática para embeddings)
    norm = np.linalg.norm(emb_avg)
    if norm > 0:
        emb_avg = emb_avg / norm

    # IP-Adapter FaceID espera tensor [batch, num_faces, 512]
    face_tensor = torch.from_numpy(emb_avg).unsqueeze(0).unsqueeze(0)   # [1, 1, 512]
    print("✓ Embedding médio calculado.\n")
    return face_tensor


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 3 — Loop de geração com 40 steps, CFG 7.5, 1024×1024
# ══════════════════════════════════════════════════════════════════════════════

def run_generation_loop(
    ip_model: IPAdapterFaceIDXL,
    face_embeds: torch.Tensor,
    prompts: list[str],
    negative_prompt: str = (
        "blurry, low quality, deformed, disfigured, bad anatomy, "
        "extra limbs, cloned face, gross proportions, low res"
    ),
    num_steps: int   = DEFAULT_STEPS,
    cfg_scale: float = DEFAULT_CFG,
    width: int       = DEFAULT_WIDTH,
    height: int      = DEFAULT_HEIGHT,
    seed: int        = 42,
    output_dir: str  = DEFAULT_OUTPUT_DIR,
    scale: float     = 0.8,   # Força do IP-Adapter (0–1); 0.8 equilibra fidelidade de identidade vs. adesão ao prompt
) -> list[str]:
    """
    Itera sobre `prompts`, gerando uma imagem 1024×1024 por entrada.

    Parâmetros
    ----------
    ip_model       : pipeline IP-Adapter FaceID SDXL
    face_embeds    : tensor de embedding médio [1, 1, 512]
    prompts        : lista de strings com os cenários a gerar
    negative_prompt: prompt negativo compartilhado por todas as gerações
    num_steps      : número de inference steps (padrão 40)
    cfg_scale      : Classifier-Free Guidance scale (padrão 7.5)
    width / height : resolução de saída (padrão 1024 × 1024)
    seed           : semente inicial; incrementada a cada imagem
    output_dir     : pasta de destino para os arquivos .png
    scale          : intensidade do ip-adapter (0.0–1.0)

    Retorna
    -------
    list[str] — caminhos dos arquivos gerados
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    generator = torch.Generator(device=DEVICE)

    for idx, prompt in enumerate(prompts):
        current_seed = seed + idx
        generator.manual_seed(current_seed)

        print(f"[{idx + 1}/{len(prompts)}] Gerando → \"{prompt[:80]}…\"")
        print(f"           seed={current_seed}  steps={num_steps}  cfg={cfg_scale}")

        image: Image.Image = ip_model.generate(
            prompt          = prompt,
            negative_prompt = negative_prompt,
            faceid_embeds   = face_embeds,
            num_inference_steps = num_steps,
            guidance_scale  = cfg_scale,
            width           = width,
            height          = height,
            scale           = scale,
            generator       = generator,
            num_samples     = 1,
        )[0]

        filename = f"output_{idx:03d}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        saved_paths.append(filepath)
        print(f"           ✓ Salvo em: {filepath}\n")

    print(f"✓ Loop concluído. {len(saved_paths)} imagem(ns) gerada(s) em '{output_dir}'.")
    return saved_paths


# ══════════════════════════════════════════════════════════════════════════════
# Ponto de entrada CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gerador SDXL + IP-Adapter FaceID com blend de dois rostos."
    )
    parser.add_argument("--face-a",   required=True,  help="Caminho da 1ª foto de rosto")
    parser.add_argument("--face-b",   required=True,  help="Caminho da 2ª foto de rosto")
    parser.add_argument(
        "--prompts", nargs="+", required=True,
        help="Um ou mais prompts de texto (use aspas para frases com espaços)",
    )
    parser.add_argument("--output",   default=DEFAULT_OUTPUT_DIR, help="Pasta de saída")
    parser.add_argument("--steps",    type=int,   default=DEFAULT_STEPS,  help="Inference steps")
    parser.add_argument("--cfg",      type=float, default=DEFAULT_CFG,    help="CFG scale")
    parser.add_argument("--seed",     type=int,   default=42,             help="Semente inicial")
    parser.add_argument("--scale",    type=float, default=0.8,            help="IP-Adapter scale")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--vae-model",  default=DEFAULT_VAE_MODEL)
    parser.add_argument("--ip-ckpt",    default=DEFAULT_IP_CKPT)
    parser.add_argument("--device",     default=DEVICE, choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Pipeline ──────────────────────────────────────────────────────────
    ip_model = build_pipeline(
        base_model = args.base_model,
        vae_model  = args.vae_model,
        ip_ckpt    = args.ip_ckpt,
        device     = args.device,
    )

    # ── 2. Embeddings faciais ─────────────────────────────────────────────────
    analyzer = build_face_analyzer(device=args.device)
    face_embeds = average_face_embeddings(
        analyzer    = analyzer,
        path_face_a = args.face_a,
        path_face_b = args.face_b,
    )
    face_embeds = face_embeds.to(args.device, dtype=torch.float16)

    # ── 3. Loop de geração ───────────────────────────────────────────────────
    run_generation_loop(
        ip_model   = ip_model,
        face_embeds= face_embeds,
        prompts    = args.prompts,
        num_steps  = args.steps,
        cfg_scale  = args.cfg,
        seed       = args.seed,
        output_dir = args.output,
        scale      = args.scale,
    )


if __name__ == "__main__":
    main()
