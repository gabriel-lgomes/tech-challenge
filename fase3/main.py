"""
Ponto de entrada do assistente médico — Fase 3.

Modos de uso:
  python main.py setup          → gera dados sintéticos e constrói índice vetorial
  python main.py ask            → modo interativo de perguntas
  python main.py flow <id>      → executa fluxo LangGraph para um paciente
  python main.py finetune       → inicia pipeline de fine-tuning
"""

import argparse
import sys
import uuid


def cmd_setup(args):
    print("=== SETUP: Gerando dados e construindo índice ===\n")

    from src.data.synthetic_generator import save_synthetic_data
    save_synthetic_data(output_dir="fase3/data/synthetic")

    from src.data.preprocessing import load_raw_dataset, build_hf_dataset, save_processed
    raw = load_raw_dataset("fase3/data/synthetic/fine_tuning_dataset.json")
    train_ds, val_ds = build_hf_dataset(raw, format="alpaca")
    save_processed(train_ds, val_ds, output_dir="fase3/data/processed")

    from src.assistant.database import init_db, seed_from_json
    init_db(db_path="fase3/data/hospital.db")
    seed_from_json(json_path="fase3/data/synthetic/patients.json", db_path="fase3/data/hospital.db")

    from src.assistant.retriever import build_index
    build_index(
        protocols_path="fase3/data/synthetic/protocols.json",
        faq_path="fase3/data/synthetic/faq.json",
        chroma_path="fase3/chroma_db",
    )
    print("\nSetup concluído. Execute 'python main.py ask' para iniciar o assistente.")


def cmd_ask(args):
    from src.assistant.chains import ask
    session_id = str(uuid.uuid4())[:8]
    print("=== ASSISTENTE MÉDICO VIRTUAL ===")
    print("Digite sua pergunta clínica (ou 'sair' para encerrar).\n")

    patient_id = input("ID do paciente (Enter para pular): ").strip() or None

    while True:
        question = input("\nPergunta: ").strip()
        if question.lower() in ("sair", "exit", "quit"):
            break
        if not question:
            continue

        result = ask(
            question=question,
            patient_id=patient_id,
            session_id=session_id,
            chroma_path="fase3/chroma_db",
        )
        print(f"\n{result['answer']}")
        if result["safety_flags"]:
            print(f"\n⚠ Flags de segurança: {', '.join(result['safety_flags'])}")


def cmd_flow(args):
    from src.graph.medical_graph import run_patient_flow
    patient_id = args.patient_id
    print(f"=== FLUXO LANGGRAPH — Paciente {patient_id} ===\n")
    summary = run_patient_flow(patient_id=patient_id, session_id=str(uuid.uuid4())[:8])
    print(summary)


def cmd_finetune(args):
    from src.data.preprocessing import load_raw_dataset, build_hf_dataset, save_processed
    raw = load_raw_dataset("fase3/data/synthetic/fine_tuning_dataset.json")
    train_ds, val_ds = build_hf_dataset(raw)
    save_processed(train_ds, val_ds, output_dir="fase3/data/processed")

    from src.fine_tuning.trainer import run_training
    run_training(
        model_name=args.model,
        dataset_dir="fase3/data/processed",
        output_dir="fase3/models/fine_tuned",
        num_epochs=args.epochs,
        max_steps=args.max_steps,
    )


def main():
    parser = argparse.ArgumentParser(description="Assistente Médico Virtual — Fase 3")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("setup", help="Gera dados sintéticos e constrói o índice vetorial")

    sub.add_parser("ask", help="Modo interativo de perguntas clínicas")

    flow_parser = sub.add_parser("flow", help="Executa fluxo LangGraph para um paciente")
    flow_parser.add_argument("patient_id", help="ID do paciente")

    ft_parser = sub.add_parser("finetune", help="Inicia pipeline de fine-tuning")
    ft_parser.add_argument("--model", default="unsloth/Phi-3-mini-4k-instruct")
    ft_parser.add_argument("--epochs", type=int, default=3)
    ft_parser.add_argument("--max-steps", type=int, default=200)

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "flow":
        cmd_flow(args)
    elif args.command == "finetune":
        cmd_finetune(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
