"""
Pipeline de fine-tuning com QLoRA usando a biblioteca Unsloth.
Projetado para rodar no Google Colab (GPU T4/A100) ou localmente com GPU >= 8GB VRAM.

Uso:
    python -m src.fine_tuning.trainer --model unsloth/Phi-3-mini-4k-instruct --epochs 3
"""

import argparse
import os
from pathlib import Path

from datasets import load_from_disk

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 200,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 10,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "output_dir": "models/fine_tuned",
    "report_to": "none",
}


def load_model_and_tokenizer(model_name: str, max_seq_len: int = 2048):
    """Carrega modelo base com quantização 4-bit via Unsloth."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
        print(f"Modelo '{model_name}' carregado via Unsloth (4-bit QLoRA)")
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig
        import torch

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        peft_config = LoraConfig(**LORA_CONFIG)
        model = get_peft_model(model, peft_config)
        print(f"Modelo '{model_name}' carregado via HF Transformers + BitsAndBytes (fallback)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_training(
    model_name: str = "unsloth/Phi-3-mini-4k-instruct",
    dataset_dir: str = "data/processed",
    output_dir: str = "models/fine_tuned",
    num_epochs: int = 3,
    max_steps: int = 200,
) -> None:
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = load_model_and_tokenizer(model_name)

    train_dataset = load_from_disk(f"{dataset_dir}/train")
    val_dataset = load_from_disk(f"{dataset_dir}/val")

    args = TRAINING_ARGS.copy()
    args["output_dir"] = output_dir
    args["max_steps"] = max_steps
    args["num_train_epochs"] = num_epochs

    training_args = TrainingArguments(**args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )

    print("Iniciando fine-tuning...")
    trainer_stats = trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nFine-tuning concluído. Modelo salvo em '{output_dir}'")
    print(f"Steps: {trainer_stats.global_step} | Loss final: {trainer_stats.training_loss:.4f}")


def evaluate_model(model_name: str, test_questions: list[str] | None = None) -> None:
    """Avaliação qualitativa — gera respostas para perguntas de teste."""
    from transformers import pipeline

    if test_questions is None:
        test_questions = [
            "Qual é a dose de AAS para prevenção secundária cardiovascular?",
            "Quando indicar hemodiálise de urgência?",
            "Qual é o protocolo de manejo da anafilaxia?",
        ]

    pipe = pipeline("text-generation", model=model_name, max_new_tokens=256)

    print("\n=== AVALIAÇÃO DO MODELO ===")
    for question in test_questions:
        prompt = (
            f"### Sistema:\nVocê é um assistente médico hospitalar especializado.\n\n"
            f"### Instrução:\nResponda à pergunta clínica.\n\n"
            f"### Entrada:\n{question}\n\n"
            f"### Resposta:\n"
        )
        result = pipe(prompt)[0]["generated_text"]
        response = result.split("### Resposta:")[-1].strip()
        print(f"\nPergunta: {question}")
        print(f"Resposta: {response[:500]}...")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning de LLM médica com QLoRA")
    parser.add_argument("--model", default="unsloth/Phi-3-mini-4k-instruct", help="Modelo base")
    parser.add_argument("--dataset-dir", default="data/processed", help="Diretório do dataset processado")
    parser.add_argument("--output-dir", default="models/fine_tuned", help="Onde salvar o modelo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-only", action="store_true", help="Apenas avaliar modelo existente")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_model(args.output_dir)
    else:
        run_training(
            model_name=args.model,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            max_steps=args.max_steps,
        )
