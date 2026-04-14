"""
train.py
────────────────────────────────────────────────────────────────────────────────
Pipeline completo de Fine-Tuning com QLoRA (Quantized Low-Rank Adaptation)
para especialização do modelo Llama 2 7B no domínio de Python e Boas Práticas.

Técnicas utilizadas:
  • QLoRA  : quantização 4-bit (NF4) + LoRA para treinar apenas ~0,1% dos pesos
  • PEFT   : Parameter-Efficient Fine-Tuning via biblioteca `peft`
  • SFTTrainer: Supervised Fine-Tuning Trainer da biblioteca `trl`

Requisitos de hardware:
  GPU com ≥ 10 GB VRAM (ex.: T4, A10, RTX 3080)
  Recomendado: Google Colab Pro (A100) ou Kaggle (T4×2)

Instalação:
    pip install -r requirements.txt

Uso:
    python train.py

Saída:
    outputs/llama2-python-lora/   ← pesos do adaptador LoRA
────────────────────────────────────────────────────────────────────────────────
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES GLOBAIS
# ══════════════════════════════════════════════════════════════════════════════

# Modelo base — versão pública sem necessidade de token do Hugging Face
BASE_MODEL = "NousResearch/Llama-2-7b-hf"

# Caminhos de dados e saída
DATASET_TRAIN = "data/train.jsonl"
DATASET_TEST  = "data/test.jsonl"
OUTPUT_DIR    = "./outputs/llama2-python-lora"

# Comprimento máximo de sequência (tokens)
MAX_SEQ_LENGTH = 512


# ══════════════════════════════════════════════════════════════════════════════
#  PASSO 2 — CONFIGURAÇÃO DA QUANTIZAÇÃO (QLoRA)
# ══════════════════════════════════════════════════════════════════════════════
# O BitsAndBytesConfig carrega o modelo base em 4-bit usando NormalFloat 4-bit
# (nf4), reduzindo o uso de VRAM de ~28 GB (fp32) para ~5 GB.
# O compute_dtype=float16 garante que as operações de forward/backward
# sejam realizadas em meia precisão, mantendo estabilidade numérica.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # Habilita quantização 4-bit
    bnb_4bit_quant_type="nf4",               # NormalFloat 4-bit (melhor para LLMs)
    bnb_4bit_compute_dtype=torch.float16,    # Dtype para operações de computação
    bnb_4bit_use_double_quant=True,          # Dupla quantização: quantiza os
                                             # próprios fatores de escala (≈0,4 GB extras)
)


# ══════════════════════════════════════════════════════════════════════════════
#  PASSO 3 — ARQUITETURA DO LoRA
# ══════════════════════════════════════════════════════════════════════════════
# O LoRA (Low-Rank Adaptation) congela a matriz de pesos original W ∈ ℝ^(d×k)
# e introduz duas matrizes treináveis de baixo rank:
#   A ∈ ℝ^(r×k)   e   B ∈ ℝ^(d×r)   onde r << min(d, k)
#
# A atualização efetiva é: ΔW = (α/r) · B·A
#
# Hiperparâmetros obrigatórios (conforme especificação do laboratório):
#   r=64   → rank das matrizes de decomposição
#   α=16   → fator de escala dos novos pesos (lora_alpha)
#   p=0.1  → dropout para regularização (evitar overfitting)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",         # Tarefa: Causal Language Modeling
    r=64,                          # Rank: dimensão das matrizes menores
    lora_alpha=16,                 # Alpha: fator de escala dos novos pesos
    lora_dropout=0.1,              # Dropout para evitar overfitting
    bias="none",                   # Não treina termos de bias
    target_modules=[               # Módulos de atenção e FFN a serem adaptados
        "q_proj",                  # Query projection
        "k_proj",                  # Key projection
        "v_proj",                  # Value projection
        "o_proj",                  # Output projection
        "gate_proj",               # Gate (SwiGLU FFN)
        "up_proj",                 # Up projection (FFN)
        "down_proj",               # Down projection (FFN)
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
#  PASSO 4 — PIPELINE DE TREINAMENTO E OTIMIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # ── Regime de treinamento ─────────────────────────────────────────────
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,       # Aumentar se VRAM insuficiente

    # ── Otimizador: paged_adamw_32bit ─────────────────────────────────────
    # Variante paginada do AdamW: transfere os estados do otimizador (momentum,
    # variância) para a RAM da CPU em picos de uso, evitando OOM na GPU.
    optim="paged_adamw_32bit",

    # ── Taxa de aprendizado ───────────────────────────────────────────────
    learning_rate=2e-4,

    # ── Scheduler: cosine ─────────────────────────────────────────────────
    # A taxa de aprendizado decai suavemente seguindo uma curva de cosseno,
    # evitando oscilações bruscas ao final do treinamento.
    lr_scheduler_type="cosine",

    # ── Warmup ratio: 0.03 ────────────────────────────────────────────────
    # Os primeiros 3% dos steps aumentam a taxa de aprendizado gradativamente
    # de 0 até learning_rate, evitando instabilidade no início.
    warmup_ratio=0.03,

    # ── Precisão mista ────────────────────────────────────────────────────
    fp16=True,
    bf16=False,                          # Usar True apenas em GPUs Ampere+

    # ── Regularização e gradientes ────────────────────────────────────────
    weight_decay=0.001,
    max_grad_norm=0.3,                   # Gradient clipping

    # ── Logging e checkpoints ─────────────────────────────────────────────
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to="tensorboard",

    # ── Agrupamento de sequências ─────────────────────────────────────────
    group_by_length=True,                # Agrupa sequências de tamanho similar
                                         # para minimizar padding e acelerar treino
)


# ══════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer() -> tuple:
    """Carrega o modelo base com quantização 4-bit e o tokenizer correspondente."""

    print(f"  Carregando tokenizer de: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    # Llama 2 não possui pad_token por padrão; usamos eos_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    # Necessário para evitar avisos com fp16

    print(f"  Carregando modelo em 4-bit (NF4 + float16 compute)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",              # Distribui automaticamente entre GPU/CPU
        trust_remote_code=True,
    )

    # Preparação obrigatória para treino com kbit (habilita gradient checkpointing
    # e converte camadas de normalização para float32)
    model = prepare_model_for_kbit_training(model)

    # Desativa cache KV (incompatível com gradient checkpointing durante treino)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer


def print_trainable_parameters(model) -> None:
    """Exibe o número de parâmetros treináveis vs. totais."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"\n  Parâmetros treináveis : {trainable:>12,}")
    print(f"  Parâmetros totais     : {total:>12,}")
    print(f"  Proporção treinável   : {pct:.4f}%\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("  Laboratório 07 — Fine-Tuning com QLoRA (PEFT + SFTTrainer)")
    print("=" * 65)

    # ── [1/5] Verificação de hardware ────────────────────────────────────
    print("\n[1/5] Verificando hardware...")
    if not torch.cuda.is_available():
        print("  [AVISO] CUDA não detectado. O treinamento em CPU é inviável.")
        print("          Use Google Colab, Kaggle ou outro ambiente com GPU.\n")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU detectada : {gpu_name}")
        print(f"  VRAM total    : {vram_gb:.1f} GB")

    # ── [2/5] Carregamento dos datasets ──────────────────────────────────
    print("\n[2/5] Carregando datasets...")
    dataset_train = load_dataset("json", data_files=DATASET_TRAIN, split="train")
    dataset_test  = load_dataset("json", data_files=DATASET_TEST,  split="train")
    print(f"  Treino : {len(dataset_train)} exemplos")
    print(f"  Teste  : {len(dataset_test)}  exemplos")

    # ── [3/5] Carregamento do modelo e tokenizer ─────────────────────────
    print("\n[3/5] Carregando modelo base com quantização 4-bit (QLoRA)...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"  ✓ Modelo carregado: {BASE_MODEL}")

    # Aplica adaptadores LoRA ao modelo quantizado
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # ── [4/5] Configuração do SFTTrainer ─────────────────────────────────
    print("[4/5] Configurando SFTTrainer...")
    print(f"  LoRA   : r={lora_config.r}, alpha={lora_config.lora_alpha}, "
          f"dropout={lora_config.lora_dropout}")
    print(f"  Optim  : {training_args.optim}")
    print(f"  LR     : {training_args.learning_rate} (scheduler: {training_args.lr_scheduler_type})")
    print(f"  Warmup : {training_args.warmup_ratio * 100:.0f}% dos steps")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        peft_config=lora_config,
        dataset_text_field="text",      # Campo do .jsonl que contém o texto completo
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,                  # Desabilitado para preservar estrutura instruction/response
    )

    # ── [5/5] Treinamento e salvamento ───────────────────────────────────
    print("\n[5/5] Iniciando treinamento...\n")
    trainer.train()

    # Salva apenas os pesos do adaptador LoRA (muito menor que o modelo completo)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n{'=' * 65}")
    print(f"  ✅ Treinamento concluído!")
    print(f"  Adaptador LoRA salvo em: {OUTPUT_DIR}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
