"""
Preprocessing e curadoria do dataset para fine-tuning.
Inclui anonimização, formatação de prompts e split treino/validação.
"""

import json
import re
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split

SYSTEM_PROMPT = (
    "Você é um assistente médico hospitalar especializado. "
    "Responda sempre com base nos protocolos internos do hospital. "
    "Nunca prescreva medicamentos diretamente — indique avaliação médica quando necessário. "
    "Cite a fonte da informação ao final de cada resposta."
)

PII_PATTERNS = [
    (r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF]"),
    (r"\b\d{2}/\d{2}/\d{4}\b", "[DATA_NASC]"),
    (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[NOME]"),
    (r"\b\d{5}-\d{3}\b", "[CEP]"),
    (r"\b\(\d{2}\)\s*\d{4,5}-\d{4}\b", "[TELEFONE]"),
]


def anonymize(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def format_alpaca(sample: dict) -> str:
    """Formata no estilo Alpaca (instruction-tuning)."""
    instruction = anonymize(sample.get("instruction", ""))
    inp = anonymize(sample.get("input", ""))
    output = anonymize(sample.get("output", ""))

    if inp:
        return (
            f"### Sistema:\n{SYSTEM_PROMPT}\n\n"
            f"### Instrução:\n{instruction}\n\n"
            f"### Entrada:\n{inp}\n\n"
            f"### Resposta:\n{output}"
        )
    return (
        f"### Sistema:\n{SYSTEM_PROMPT}\n\n"
        f"### Instrução:\n{instruction}\n\n"
        f"### Resposta:\n{output}"
    )


def format_chat(sample: dict) -> dict:
    """Formata no estilo ChatML para modelos que usam chat templates."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{sample['instruction']}\n\n{sample.get('input', '')}".strip()},
        {"role": "assistant", "content": sample["output"]},
    ]
    return {"messages": messages}


def load_raw_dataset(path: str = "data/synthetic/fine_tuning_dataset.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_hf_dataset(
    raw_data: list[dict],
    format: str = "alpaca",
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Retorna (train_dataset, val_dataset) no formato Hugging Face.
    """
    if format == "alpaca":
        texts = [{"text": format_alpaca(s)} for s in raw_data]
        train_raw, val_raw = train_test_split(texts, test_size=val_size, random_state=seed)
        return Dataset.from_list(train_raw), Dataset.from_list(val_raw)

    if format == "chat":
        chats = [format_chat(s) for s in raw_data]
        train_raw, val_raw = train_test_split(chats, test_size=val_size, random_state=seed)
        return Dataset.from_list(train_raw), Dataset.from_list(val_raw)

    raise ValueError(f"Formato desconhecido: {format}. Use 'alpaca' ou 'chat'.")


def save_processed(train_ds: Dataset, val_ds: Dataset, output_dir: str = "data/processed") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(f"{output_dir}/train")
    val_ds.save_to_disk(f"{output_dir}/val")
    print(f"Dataset processado salvo em '{output_dir}/'")
    print(f"  Treino: {len(train_ds)} amostras | Validação: {len(val_ds)} amostras")


if __name__ == "__main__":
    raw = load_raw_dataset()
    train_ds, val_ds = build_hf_dataset(raw, format="alpaca")
    save_processed(train_ds, val_ds)
