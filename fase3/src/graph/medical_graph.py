"""
Fluxo de decisão médica automatizado com LangGraph.

Fluxo:
  receber_paciente → verificar_exames → sugerir_tratamento → emitir_alertas → finalizar

Cada nó é uma etapa independente que pode chamar o assistente LangChain.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

from src.assistant.chains import ask
from src.assistant.database import get_patient, get_pending_exams, add_alert
from src.security.audit_logger import AuditLogger

logger = AuditLogger()


class PatientState(TypedDict):
    patient_id: str
    session_id: str
    patient_data: dict
    pending_exams: list[str]
    exam_recommendations: str
    treatment_suggestion: str
    alerts_emitted: list[str]
    summary: str


# --- Nós do grafo ---

def node_receive_patient(state: PatientState) -> PatientState:
    """Carrega dados do paciente do banco de dados."""
    patient = get_patient(state["patient_id"])
    if not patient:
        return {**state, "patient_data": {}, "summary": "Paciente não encontrado."}
    print(f"[GRAFO] Paciente carregado: {state['patient_id']} — {patient['diagnosis']}")
    return {**state, "patient_data": patient}


def node_check_exams(state: PatientState) -> PatientState:
    """Verifica exames pendentes e consulta o assistente sobre prioridade."""
    if not state.get("patient_data"):
        return {**state, "pending_exams": [], "exam_recommendations": "Sem dados de paciente."}

    exams = get_pending_exams(state["patient_id"])
    if not exams:
        return {**state, "pending_exams": [], "exam_recommendations": "Nenhum exame pendente."}

    exam_list = ", ".join(exams)
    question = (
        f"O paciente com diagnóstico de '{state['patient_data']['diagnosis']}' "
        f"possui os seguintes exames pendentes: {exam_list}. "
        f"Quais devem ser priorizados e por quê?"
    )
    result = ask(question, patient_id=state["patient_id"], session_id=state["session_id"])
    print(f"[GRAFO] Exames avaliados: {exam_list}")
    return {**state, "pending_exams": exams, "exam_recommendations": result["answer"]}


def node_suggest_treatment(state: PatientState) -> PatientState:
    """Sugere condutas com base no diagnóstico e nos protocolos internos."""
    if not state.get("patient_data"):
        return {**state, "treatment_suggestion": "Sem dados suficientes."}

    p = state["patient_data"]
    question = (
        f"Com base no diagnóstico de '{p['diagnosis']}' e nas medicações em uso "
        f"({', '.join(p.get('medications', []))}), quais condutas o protocolo hospitalar recomenda?"
    )
    result = ask(question, patient_id=state["patient_id"], session_id=state["session_id"])
    print(f"[GRAFO] Sugestão de tratamento gerada.")
    return {**state, "treatment_suggestion": result["answer"]}


def node_emit_alerts(state: PatientState) -> PatientState:
    """Emite alertas críticos com base nos dados do paciente."""
    alerts = []
    p = state.get("patient_data", {})

    if "pendentes" in state.get("exam_recommendations", "").lower():
        alert = f"ALERTA: Exames pendentes críticos detectados para o paciente {state['patient_id']}."
        alerts.append(alert)
        add_alert(state["patient_id"], alert)

    existing = p.get("alerts", [])
    for a in existing:
        alerts.append(f"ALERTA (prontuário): {a}")

    for alert in alerts:
        print(f"[GRAFO] {alert}")
        logger.log(
            session_id=state["session_id"],
            question="[Sistema] Emissão automática de alerta",
            answer=alert,
            sources=["Sistema de alertas automáticos"],
            safety_flags=[],
            node="emit_alerts",
            patient_id=state["patient_id"],
        )

    return {**state, "alerts_emitted": alerts}


def node_finalize(state: PatientState) -> PatientState:
    """Monta o resumo final do fluxo."""
    p = state.get("patient_data", {})
    summary = (
        f"=== RESUMO DO FLUXO — Paciente {state['patient_id']} ===\n"
        f"Diagnóstico: {p.get('diagnosis', 'N/A')}\n"
        f"Ala: {p.get('ward', 'N/A')}\n\n"
        f"[Exames pendentes]\n{state.get('exam_recommendations', '')}\n\n"
        f"[Sugestão de conduta]\n{state.get('treatment_suggestion', '')}\n\n"
        f"[Alertas emitidos]\n" + "\n".join(state.get("alerts_emitted", ["Nenhum"])) +
        "\n\n[Aviso] Todas as sugestões requerem validação do médico responsável."
    )
    return {**state, "summary": summary}


# --- Construção do grafo ---

def build_medical_graph() -> StateGraph:
    graph = StateGraph(PatientState)

    graph.add_node("receber_paciente", node_receive_patient)
    graph.add_node("verificar_exames", node_check_exams)
    graph.add_node("sugerir_tratamento", node_suggest_treatment)
    graph.add_node("emitir_alertas", node_emit_alerts)
    graph.add_node("finalizar", node_finalize)

    graph.set_entry_point("receber_paciente")
    graph.add_edge("receber_paciente", "verificar_exames")
    graph.add_edge("verificar_exames", "sugerir_tratamento")
    graph.add_edge("sugerir_tratamento", "emitir_alertas")
    graph.add_edge("emitir_alertas", "finalizar")
    graph.add_edge("finalizar", END)

    return graph.compile()


def run_patient_flow(patient_id: str, session_id: str = "graph-session") -> str:
    """Executa o fluxo completo para um paciente e retorna o resumo."""
    app = build_medical_graph()
    initial_state: PatientState = {
        "patient_id": patient_id,
        "session_id": session_id,
        "patient_data": {},
        "pending_exams": [],
        "exam_recommendations": "",
        "treatment_suggestion": "",
        "alerts_emitted": [],
        "summary": "",
    }
    final_state = app.invoke(initial_state)
    return final_state["summary"]
