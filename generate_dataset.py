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
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Configurações ─────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Modelos gratuitos disponíveis no OpenRouter (verificados em Abril/2025)
FREE_MODELS = [
    "openrouter/auto",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

TARGET_PAIRS = 55      # Mínimo exigido: 50; geramos 55 para margem de segurança
TRAIN_RATIO  = 0.90    # 90% treino / 10% teste
DATA_DIR     = "data"
RETRY_LIMIT  = 3
RETRY_DELAY  = 15      # segundos entre tentativas (aumentado para respeitar rate limit)

# Tópicos que cobrem o domínio Python de forma abrangente
TOPICS = [
    "estruturas de dados nativas do Python (listas, dicionários, conjuntos e tuplas)",
    "funções, closures e escopo de variáveis",
    "programação orientada a objetos: classes, herança e polimorfismo",
    "tratamento de exceções com try/except/finally",
    "list comprehensions, dict comprehensions e expressões geradoras",
    "decorators e metaprogramação em Python",
    "context managers e o protocolo __enter__ e __exit__",
    "manipulação de arquivos e caminhos com pathlib",
    "testes unitários com pytest e fixtures",
    "programação assíncrona com asyncio e async/await",
    "expressões regulares com o módulo re",
    "consumo de APIs REST com a biblioteca requests",
    "boas práticas, PEP 8 e type hints",
    "módulos, pacotes e gerenciamento de dependências com pip e virtualenv",
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


# ── Parsing robusto de JSON ───────────────────────────────────────────────────

def _clean_and_repair_json(raw: str) -> str:
    """
    Limpa e tenta reparar o JSON retornado pelo modelo.

    Problemas tratados:
    1. Blocos de markdown (```json ... ```)
    2. Escapes inválidos dentro de strings (ex: \\d → \\\\d em regex)
    3. JSON truncado — tenta extrair pares completos mesmo sem o fechamento
    """
    raw = raw.strip()

    # 1. Remove blocos de markdown
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())
    raw = raw.strip()

    # 2. Tenta parsear diretamente (caminho feliz)
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # 3. Corrige escapes inválidos comuns gerados por código Python em strings JSON
    #    Padrão: backslash seguido de letra que não é escape JSON válido
    #    Escapes válidos: \" \\ \/ \b \f \n \r \t \uXXXX
    def fix_invalid_escapes(s: str) -> str:
        return re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', s)

    repaired = fix_invalid_escapes(raw)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    # 4. JSON truncado: extrai pares individuais com regex
    #    Captura objetos {"prompt": "...", "response": "..."} completos
    pairs_raw = re.findall(
        r'\{\s*"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"response"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        raw,
        re.DOTALL,
    )
    if pairs_raw:
        pairs = [{"prompt": p, "response": r} for p, r in pairs_raw]
        return json.dumps({"pairs": pairs}, ensure_ascii=False)

    return raw  # Retorna original para que o JSONDecodeError seja relançado


def safe_parse(raw: str) -> list[dict]:
    """Tenta parsear o JSON e retorna a lista de pares, ou lista vazia em falha."""
    try:
        cleaned = _clean_and_repair_json(raw)
        data = json.loads(cleaned)
        return data.get("pairs", [])
    except (json.JSONDecodeError, AttributeError):
        return []


# ── Geração de pares por tópico ───────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Você é um engenheiro de software sênior especializado em Python. "
    "Suas respostas são precisas, didáticas e incluem exemplos de código "
    "quando pertinente. "
    "IMPORTANTE: Retorne APENAS JSON válido, sem texto antes ou depois."
)

# Prompt mais simples e com instrução explícita sobre escapes — reduz truncamentos
USER_TEMPLATE = """Gere {n} pares distintos de instrução e resposta sobre: "{topic}".

REGRAS OBRIGATORIAS:
- Cada instrução deve ser uma pergunta ou tarefa prática e realista.
- Cada resposta deve ter explicação conceitual e exemplo de código.
- Varie o estilo: conceitual, código, depuração, comparação.
- Mantenha respostas com no máximo 400 palavras cada para evitar truncamento.
- Em exemplos de código dentro do JSON, use aspas simples em vez de duplas.
- Em padrões regex dentro do JSON, escreva as barras duplas: \\\\d, \\\\w, \\\\s etc.

Retorne SOMENTE o JSON abaixo, sem markdown, sem texto extra:
{{
  "pairs": [
    {{
      "prompt": "instrução ou pergunta do usuário",
      "response": "resposta técnica detalhada com exemplo de código"
    }}
  ]
}}"""


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
                    temperature=0.7,
                    max_tokens=3500,
                )
                raw = response.choices[0].message.content
                pairs = safe_parse(raw)
                if pairs:
                    return pairs
                print(f"    [AVISO] JSON inválido após reparo (modelo={model}, tentativa={attempt})")
            except Exception as e:
                err_str = str(e)
                # Rate limit: aguarda mais antes de tentar novamente
                if "429" in err_str:
                    print(f"    [AVISO] Rate limit (modelo={model}, tentativa={attempt}). Aguardando {RETRY_DELAY*2}s...")
                    time.sleep(RETRY_DELAY * 2)
                elif "404" in err_str:
                    print(f"    [AVISO] Modelo não disponível: {model}. Pulando.")
                    break  # Não adianta tentar de novo
                else:
                    print(f"    [AVISO] Erro na API (modelo={model}, tentativa={attempt}): {e}")

            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

        print(f"    [INFO] Modelo '{model}' esgotou tentativas. Tentando próximo modelo...")

    return []


# ── Orquestração principal ────────────────────────────────────────────────────

def generate_dataset(client: OpenAI, target: int = TARGET_PAIRS) -> list[dict]:
    all_pairs: list[dict] = []
    topics = TOPICS.copy()
    random.shuffle(topics)
    topic_idx = 0
    consecutive_failures = 0

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
            consecutive_failures = 0
            print(f"          ✓ {len(pairs)} par(es) recebido(s).")
        else:
            consecutive_failures += 1
            print(f"          ✗ Falha. Pulando tópico.")
            # Se falhar 5 vezes seguidas, aguarda mais antes de continuar
            if consecutive_failures >= 5:
                print(f"\n  [INFO] {consecutive_failures} falhas consecutivas. Aguardando 60s para reset de rate limit...\n")
                time.sleep(60)
                consecutive_failures = 0

        time.sleep(2)  # Pausa entre requisições

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
    split_idx  = int(len(pairs) * train_ratio)
    train_data = pairs[:split_idx]
    test_data  = pairs[split_idx:]

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
