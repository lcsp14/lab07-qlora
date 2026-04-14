"""
generate_dataset.py
────────────────────────────────────────────────────────────────────────────────
Script para geração de dataset sintético de instruções no domínio de
"Assistente Especializado em Python e Boas Práticas de Programação".

Utiliza a API do OpenRouter (compatível com a interface OpenAI) com modelos
gratuitos para gerar, no mínimo, 55 pares de prompt/response.

Requisitos:
    pip install openai python-dotenv

Configuração:
    Crie um arquivo .env na raiz do projeto com:
    OPENROUTER_API_KEY=sk-or-v1-...

Uso:
    python generate_dataset.py

Saída:
    data/train.jsonl  (90% dos pares)
    data/test.jsonl   (10% dos pares)
────────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import random
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Configurações ─────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Modelos gratuitos disponíveis no OpenRouter (fallback em cadeia)
FREE_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "openchat/openchat-7b:free",
]

TARGET_PAIRS = 55      # Mínimo exigido: 50; geramos 55 para margem de segurança
TRAIN_RATIO  = 0.90    # 90% treino / 10% teste
DATA_DIR     = "data"
RETRY_LIMIT  = 3
RETRY_DELAY  = 5       # segundos entre tentativas

# Tópicos que cobrem o domínio Python de forma abrangente
TOPICS = [
    "estruturas de dados nativas do Python (listas, dicionários, conjuntos e tuplas)",
    "funções, closures e escopo de variáveis",
    "programação orientada a objetos: classes, herança e polimorfismo",
    "tratamento de exceções com try/except/finally",
    "list comprehensions, dict comprehensions e expressões geradoras",
    "decorators e metaprogramação em Python",
    "context managers e o protocolo __enter__/__exit__",
    "manipulação de arquivos e caminhos com pathlib",
    "testes unitários com pytest e fixtures",
    "programação assíncrona com asyncio e async/await",
    "expressões regulares com o módulo re",
    "consumo de APIs REST com a biblioteca requests",
    "boas práticas, PEP 8 e type hints",
    "módulos, pacotes e gerenciamento de dependências com pip/virtualenv",
    "manipulação de dados com a biblioteca padrão: collections, itertools, functools",
]


# ── Cliente OpenRouter ────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "[ERRO] Variável de ambiente OPENROUTER_API_KEY não encontrada.\n"
            "       Crie um arquivo .env com: OPENROUTER_API_KEY=sk-or-v1-..."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


# ── Geração de pares por tópico ───────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Você é um engenheiro de software sênior especializado em Python. "
    "Suas respostas são precisas, didáticas e incluem exemplos de código "
    "quando pertinente."
)

USER_TEMPLATE = """Gere {n} pares distintos de instrução e resposta sobre: "{topic}".

REGRAS:
- Cada instrução deve ser uma pergunta ou tarefa prática e realista.
- Cada resposta deve ser completa, com explicação conceitual E exemplo de código.
- Varie o estilo: perguntas conceituais, pedidos de código, depuração de erros, comparações.

Retorne SOMENTE um JSON válido, sem markdown e sem texto extra, no formato:
{{
  "pairs": [
    {{
      "prompt": "instrução ou pergunta do usuário",
      "response": "resposta técnica detalhada com exemplo de código"
    }}
  ]
}}"""


def _clean_json(raw: str) -> str:
    """Remove blocos de markdown que alguns modelos inserem ao redor do JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove primeira e última linha (``` e ```)
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return raw.strip()


def generate_pairs(client: OpenAI, topic: str, n: int = 4) -> list[dict]:
    """Gera n pares de prompt/response para um tópico. Tenta modelos em cascata."""
    user_msg = USER_TEMPLATE.format(n=n, topic=topic)

    for model in FREE_MODELS:
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.75,
                    max_tokens=3000,
                )
                raw = response.choices[0].message.content
                data = json.loads(_clean_json(raw))
                pairs = data.get("pairs", [])
                if pairs:
                    return pairs
            except json.JSONDecodeError as e:
                print(f"    [AVISO] JSON inválido (modelo={model}, tentativa={attempt}): {e}")
            except Exception as e:
                print(f"    [AVISO] Erro na API (modelo={model}, tentativa={attempt}): {e}")

            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

        print(f"    [INFO] Modelo '{model}' esgotou tentativas. Tentando próximo modelo...")

    return []   # Todos os modelos falharam para este tópico


# ── Orquestração principal ────────────────────────────────────────────────────

def generate_dataset(client: OpenAI, target: int = TARGET_PAIRS) -> list[dict]:
    all_pairs: list[dict] = []
    topics = TOPICS.copy()
    random.shuffle(topics)
    topic_idx = 0

    print(f"\nMeta: {target} pares | Domínio: Python e Boas Práticas\n")

    while len(all_pairs) < target:
        topic = topics[topic_idx % len(topics)]
        topic_idx += 1
        remaining = target - len(all_pairs)
        batch_size = min(4, remaining)

        print(f"  [{len(all_pairs):>3}/{target}] Gerando {batch_size} par(es) sobre: {topic[:60]}...")
        pairs = generate_pairs(client, topic, n=batch_size)

        if pairs:
            all_pairs.extend(pairs[:batch_size])
            print(f"          ✓ {len(pairs)} par(es) recebido(s).")
        else:
            print(f"          ✗ Falha. Pulando tópico.")

        # Pequena pausa para respeitar rate limits
        time.sleep(1)

    return all_pairs[:target]


def format_record(pair: dict) -> dict:
    """
    Formata o par no template de instrução utilizado durante o fine-tuning.
    O campo 'text' é o que o SFTTrainer lerá.
    """
    return {
        "prompt":   pair["prompt"],
        "response": pair["response"],
        "text": (
            "### Instrução:\n"
            f"{pair['prompt'].strip()}\n\n"
            "### Resposta:\n"
            f"{pair['response'].strip()}"
        ),
    }


def split_and_save(pairs: list[dict], train_ratio: float = TRAIN_RATIO) -> None:
    random.shuffle(pairs)
    split_idx   = int(len(pairs) * train_ratio)
    train_data  = pairs[:split_idx]
    test_data   = pairs[split_idx:]

    os.makedirs(DATA_DIR, exist_ok=True)

    splits = {"train": train_data, "test": test_data}
    for name, data in splits.items():
        path = os.path.join(DATA_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for pair in data:
                f.write(json.dumps(format_record(pair), ensure_ascii=False) + "\n")
        print(f"  Salvo → {path}  ({len(data)} exemplos)")


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Laboratório 07 — Geração de Dataset Sintético (OpenRouter)")
    print("=" * 65)

    client = get_client()
    pairs  = generate_dataset(client)

    print(f"\nTotal gerado: {len(pairs)} pares\n")
    print("Salvando e dividindo o dataset...")
    split_and_save(pairs)

    train_count = int(len(pairs) * TRAIN_RATIO)
    test_count  = len(pairs) - train_count
    print(f"\n✅ Concluído! Divisão: {train_count} treino | {test_count} teste")
    print(f"   Arquivos: {DATA_DIR}/train.jsonl  e  {DATA_DIR}/test.jsonl")


if __name__ == "__main__":
    main()
