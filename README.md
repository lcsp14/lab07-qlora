# Lab 07 — Especialização de LLMs com LoRA e QLoRA

[![CI](https://github.com/lcsp14/lab07-qlora/actions/workflows/ci.yml/badge.svg)](https://github.com/<seu-usuario>/lab07-qlora/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![Release](https://img.shields.io/badge/release-v1.0-orange)

> **Instituto de Ensino Superior iCEV — Disciplina de LLMs (P2)**
> Pipeline completo de *fine-tuning* do Llama 2 7B no domínio de
> **Python e Boas Práticas de Programação**, usando PEFT/LoRA e quantização 4-bit.

---

## 📋 Sumário

1. [Visão Geral](#visão-geral)
2. [Arquitetura e Técnicas](#arquitetura-e-técnicas)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Configuração do Ambiente](#configuração-do-ambiente)
5. [Passo a Passo de Execução](#passo-a-passo-de-execução)
6. [Hiperparâmetros Obrigatórios](#hiperparâmetros-obrigatórios)
7. [Dataset Sintético](#dataset-sintético)
8. [Resultados Esperados](#resultados-esperados)
9. [Uso de IA — Declaração Obrigatória](#uso-de-ia--declaração-obrigatória)
10. [Referências](#referências)

---

## Visão Geral

Este projeto implementa um pipeline completo de **Quantized Low-Rank Adaptation (QLoRA)** para especializar o modelo `NousResearch/Llama-2-7b-hf` no domínio de Python, tornando o treinamento viável em GPUs com ≥ 10 GB VRAM sem abrir mão de qualidade.

| Componente         | Tecnologia                          |
|--------------------|-------------------------------------|
| Modelo Base        | Llama 2 7B (NousResearch, público)  |
| Quantização        | BitsAndBytes — NF4 4-bit + float16  |
| Adaptação          | LoRA via `peft` (r=64, α=16)        |
| Treinamento        | SFTTrainer (`trl`)                  |
| Geração de Dados   | OpenRouter API (modelos gratuitos)  |
| Monitoramento      | TensorBoard                         |

---

## Arquitetura e Técnicas

### QLoRA — Quantized Low-Rank Adaptation

O treinamento tradicional (*Full Fine-Tuning*) do Llama 2 7B exigiria ~112 GB de VRAM (pesos em bf16 + estados do otimizador). O QLoRA resolve isso em duas etapas:

```
┌─────────────────────────────────────────────────────────────┐
│                   MODELO BASE (congelado)                    │
│              W ∈ ℝ^(d×k)  — carregado em NF4 4-bit          │
│                       ↓ (frozen)                             │
│   ┌──────────────────────────────────────────────────────┐   │
│   │            ADAPTADORES LoRA (treináveis)             │   │
│   │    ΔW = (α/r) · B · A                               │   │
│   │    A ∈ ℝ^(r×k)    B ∈ ℝ^(d×r)    r = 64            │   │
│   └──────────────────────────────────────────────────────┘   │
│                       ↓                                      │
│              Saída = Wx + ΔWx                                │
└─────────────────────────────────────────────────────────────┘
```

**Resultado:** de ~112 GB → ~6 GB de VRAM, com perda mínima de qualidade.

### Módulos Adaptados

Os adaptadores LoRA são inseridos em todas as projeções de atenção e FFN:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

---

## Estrutura do Projeto

```
lab07-qlora/
│
├── generate_dataset.py      # Passo 1: geração de dataset sintético via OpenRouter
├── train.py                 # Passos 2-4: pipeline completo de treinamento QLoRA
├── requirements.txt         # Dependências Python com versões fixadas
├── .env.example             # Modelo do arquivo de variáveis de ambiente
├── .gitignore               # Arquivos e pastas ignorados pelo git
│
├── data/
│   ├── train.jsonl          # 90% dos pares (gerado por generate_dataset.py)
│   └── test.jsonl           # 10% dos pares (gerado por generate_dataset.py)
│
└── outputs/
    └── llama2-python-lora/  # Adaptador LoRA salvo após treinamento
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── tokenizer/
```

---

## Configuração do Ambiente

### Pré-requisitos

- Python 3.10+
- GPU com ≥ 10 GB VRAM e CUDA 12.x (recomendado: Google Colab Pro ou Kaggle)
- Conta gratuita no [OpenRouter](https://openrouter.ai/) para geração do dataset

### 1. Clone o repositório e entre na pasta

```bash
git clone https://github.com/<seu-usuario>/lab07-qlora.git
cd lab07-qlora
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
# Edite o arquivo .env e insira sua OPENROUTER_API_KEY
```

---

## Passo a Passo de Execução

### Passo 1 — Gerar o Dataset Sintético

```bash
python generate_dataset.py
```

O script irá:
- Conectar à API do OpenRouter usando modelos gratuitos (ex.: `mistralai/mistral-7b-instruct:free`)
- Gerar ≥ 55 pares de prompt/response no domínio de Python
- Dividir automaticamente em 90% treino e 10% teste
- Salvar em `data/train.jsonl` e `data/test.jsonl`

**Formato do .jsonl gerado:**
```json
{
  "prompt": "Como funciona um decorator em Python?",
  "response": "Um decorator é uma função que... [resposta detalhada]",
  "text": "### Instrução:\nComo funciona um decorator...\n\n### Resposta:\nUm decorator é..."
}
```

### Passo 2, 3 e 4 — Treinamento com QLoRA

```bash
python train.py
```

O script executa automaticamente:
1. Verificação de hardware (GPU/VRAM)
2. Carregamento dos datasets `.jsonl`
3. Carregamento do modelo base com quantização 4-bit (NF4)
4. Aplicação dos adaptadores LoRA (r=64, α=16, dropout=0.1)
5. Treinamento com SFTTrainer + `paged_adamw_32bit` + scheduler `cosine`
6. Salvamento do adaptador em `outputs/llama2-python-lora/`

### Monitorar o treinamento com TensorBoard

```bash
tensorboard --logdir outputs/llama2-python-lora/logs
# Acesse: http://localhost:6006
```

---

## Hiperparâmetros Obrigatórios

### Quantização (Passo 2)

| Parâmetro              | Valor     |
|------------------------|-----------|
| `load_in_4bit`         | `True`    |
| `bnb_4bit_quant_type`  | `"nf4"`   |
| `bnb_4bit_compute_dtype` | `float16` |

### LoRA (Passo 3)

| Parâmetro       | Valor          |
|-----------------|----------------|
| `task_type`     | `CAUSAL_LM`    |
| `r` (Rank)      | `64`           |
| `lora_alpha`    | `16`           |
| `lora_dropout`  | `0.1`          |

### Treinamento (Passo 4)

| Parâmetro              | Valor              |
|------------------------|--------------------|
| `optim`                | `paged_adamw_32bit`|
| `lr_scheduler_type`    | `cosine`           |
| `warmup_ratio`         | `0.03`             |
| `learning_rate`        | `2e-4`             |
| `num_train_epochs`     | `3`                |
| `per_device_train_batch_size` | `4`        |

---

## Dataset Sintético

**Domínio escolhido:** Assistente especializado em Python e Boas Práticas de Programação

**Tópicos cobertos:**
- Estruturas de dados nativas (listas, dicionários, sets, tuplas)
- Funções, closures e escopo
- Programação Orientada a Objetos (herança, polimorfismo)
- Tratamento de exceções
- List/Dict comprehensions e generators
- Decorators e metaprogramação
- Context managers
- Manipulação de arquivos com `pathlib`
- Testes com `pytest`
- Programação assíncrona (`asyncio`)
- Type hints e PEP 8
- Expressões regulares, `requests`, `collections`, `itertools`

**Distribuição:**
```
Total de pares : ≥ 55
Treino         : ≥ 49 exemplos  (90%)
Teste          : ≥  6 exemplos  (10%)
Formato        : .jsonl (JSON Lines)
```

---

## Resultados Esperados

Após o treinamento, o adaptador LoRA permite que o modelo base responda a perguntas técnicas de Python com:
- Explicações conceituais precisas
- Exemplos de código funcionais e idiomáticos
- Boas práticas e dicas de PEP 8
- Comparação entre abordagens

Para inferência, o adaptador pode ser carregado com:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base, "./outputs/llama2-python-lora")
tokenizer = AutoTokenizer.from_pretrained("./outputs/llama2-python-lora")

prompt = "### Instrução:\nO que é um generator em Python?\n\n### Resposta:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Uso de IA — Declaração Obrigatória

> **Partes geradas/complementadas com IA, revisadas por Lucas César.**

Especificamente:

- **Geração do dataset** (`generate_dataset.py`): a API do OpenRouter (modelo `mistralai/mistral-7b-instruct:free`) foi utilizada para gerar os pares de instrução/resposta. Todo o conteúdo gerado foi revisado manualmente para verificar correção técnica, coerência e alinhamento com as boas práticas de Python.
- **Templates de código**: trechos de código foram verificados e testados manualmente.
- **Revisão crítica**: todos os hiperparâmetros, configurações de quantização e arquitetura LoRA foram compreendidos, justificados e implementados de forma autônoma, seguindo a literatura de referência (QLoRA: Dettmers et al., 2023).

---

## Referências

- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Hugging Face. *PEFT Documentation*. https://huggingface.co/docs/peft
- Hugging Face. *TRL SFTTrainer*. https://huggingface.co/docs/trl/sft_trainer
- OpenRouter. *Free Models Documentation*. https://openrouter.ai/docs

---

<p align="center">
  <strong>Instituto de Ensino Superior iCEV</strong><br>
  Laboratório 07 — P2 | Release <code>v1.0</code>
</p>
