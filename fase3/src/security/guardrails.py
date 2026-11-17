"""
Guardrails de segurança: detecta perguntas que excedem os limites do assistente
e bloqueia prescrições diretas, diagnósticos definitivos sem validação humana, etc.
"""

import re

BLOCKED_PATTERNS = [
    (r"\bprescrev[ae]\b", "Prescrição direta não é permitida. Consulte o médico responsável."),
    (r"\bdose\s+letal\b", "Informação não disponível por razões de segurança."),
    (r"\bautomaticar\b.*\btratamento\b", "Decisões terapêuticas requerem validação médica."),
    (r"\bdiagnósti(co|que)\s+definitivo\b", "O assistente apoia decisões clínicas — não substitui avaliação médica."),
]

HIGH_RISK_TERMS = [
    "prescrever", "prescreva", "receitar", "administrar sem supervisão",
    "diagnóstico final", "confirmo que é", "certamente é",
]

DISCLAIMER = (
    "\n\n---\n"
    "**Aviso:** Esta resposta tem carater informativo e de apoio clínico. "
    "Decisões terapêuticas e diagnósticas devem ser validadas pelo médico responsável. "
    "Fonte indicada acima."
)


def check_input(question: str) -> tuple[bool, str]:
    """
    Verifica se a pergunta viola alguma restrição de segurança.
    Retorna (is_blocked, reason).
    """
    q_lower = question.lower()
    for pattern, reason in BLOCKED_PATTERNS:
        if re.search(pattern, q_lower):
            return True, reason
    return False, ""


def add_safety_flags(answer: str) -> list[str]:
    """Detecta flags de risco na resposta gerada."""
    flags = []
    answer_lower = answer.lower()
    for term in HIGH_RISK_TERMS:
        if term in answer_lower:
            flags.append(f"high_risk_term: '{term}'")
    return flags


def apply_disclaimer(answer: str, sources: list[str]) -> str:
    """Adiciona disclaimer e lista de fontes à resposta."""
    source_text = ""
    if sources:
        source_text = "\n\n**Fontes consultadas:**\n" + "\n".join(f"- {s}" for s in sources)
    return answer + source_text + DISCLAIMER
