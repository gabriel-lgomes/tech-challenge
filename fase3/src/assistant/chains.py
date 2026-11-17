"""
Pipeline LangChain do assistente médico.
Combina RAG (ChromaDB) + contexto do paciente (SQLite) + LLM fine-tunada.
"""

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import HuggingFacePipeline

from src.assistant.retriever import retrieve, format_context
from src.assistant.database import get_patient
from src.security.guardrails import check_input, add_safety_flags, apply_disclaimer
from src.security.audit_logger import AuditLogger

load_dotenv()

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["system_context", "patient_context", "question"],
    template=(
        "### Sistema:\n"
        "Você é um assistente médico hospitalar. Responda com base nos protocolos internos. "
        "Nunca prescreva diretamente sem validação humana.\n\n"
        "### Protocolos e conhecimento base:\n{system_context}\n\n"
        "### Contexto do paciente:\n{patient_context}\n\n"
        "### Pergunta:\n{question}\n\n"
        "### Resposta:\n"
    ),
)


def _build_llm(model_path: str | None = None):
    """Carrega a LLM fine-tunada ou usa Groq como fallback via API."""
    model_path = model_path or os.getenv("MODEL_PATH", "models/fine_tuned")

    # Tenta carregar modelo local fine-tunado
    import os.path
    if os.path.isdir(model_path) and any(
        f.endswith((".bin", ".safetensors")) for f in os.listdir(model_path)
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.1)
        return HuggingFacePipeline(pipeline=pipe)

    # Fallback: Groq API (não requer GPU)
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=os.getenv("GROQ_API_KEY"))


def build_chain(model_path: str | None = None, chroma_path: str = "chroma_db"):
    llm = _build_llm(model_path)
    logger = AuditLogger()

    def retrieve_context(inputs: dict) -> dict:
        docs = retrieve(inputs["question"], n_results=3, chroma_path=chroma_path)
        context, sources = format_context(docs)
        return {**inputs, "system_context": context, "_sources": sources}

    def add_patient_context(inputs: dict) -> dict:
        patient_id = inputs.get("patient_id")
        if patient_id:
            patient = get_patient(patient_id)
            if patient:
                ctx = (
                    f"Paciente: {patient_id} | Idade: {patient['age']} | Sexo: {patient['sex']}\n"
                    f"Diagnóstico: {patient['diagnosis']} | Ala: {patient['ward']}\n"
                    f"Medicações: {', '.join(patient['medications'])}\n"
                    f"Exames pendentes: {', '.join(patient['pending_exams'])}\n"
                    f"Alertas: {', '.join(patient['alerts'])}"
                )
                return {**inputs, "patient_context": ctx}
        return {**inputs, "patient_context": "Nenhum paciente específico informado."}

    def run_llm(inputs: dict) -> dict:
        prompt = PROMPT_TEMPLATE.format(
            system_context=inputs["system_context"],
            patient_context=inputs["patient_context"],
            question=inputs["question"],
        )
        raw_answer = llm.invoke(prompt)
        if hasattr(raw_answer, "content"):
            raw_answer = raw_answer.content
        raw_answer = str(raw_answer).split("### Resposta:")[-1].strip()
        return {**inputs, "raw_answer": raw_answer}

    def apply_safety(inputs: dict) -> dict:
        flags = add_safety_flags(inputs["raw_answer"])
        answer = apply_disclaimer(inputs["raw_answer"], inputs.get("_sources", []))
        logger.log(
            session_id=inputs.get("session_id", "unknown"),
            question=inputs["question"],
            answer=answer,
            sources=inputs.get("_sources", []),
            safety_flags=flags,
            patient_id=inputs.get("patient_id"),
        )
        return {"answer": answer, "sources": inputs.get("_sources", []), "safety_flags": flags}

    chain = (
        RunnableLambda(retrieve_context)
        | RunnableLambda(add_patient_context)
        | RunnableLambda(run_llm)
        | RunnableLambda(apply_safety)
    )
    return chain


def ask(
    question: str,
    patient_id: str | None = None,
    session_id: str = "default",
    model_path: str | None = None,
    chroma_path: str = "chroma_db",
) -> dict:
    """Interface principal do assistente."""
    is_blocked, reason = check_input(question)
    if is_blocked:
        return {"answer": f"Pergunta bloqueada por política de segurança: {reason}", "sources": [], "safety_flags": ["blocked"]}

    chain = build_chain(model_path=model_path, chroma_path=chroma_path)
    return chain.invoke({"question": question, "patient_id": patient_id, "session_id": session_id})
