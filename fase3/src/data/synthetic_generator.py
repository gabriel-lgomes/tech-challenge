"""
Gerador de dados sintéticos médicos para simular o dataset interno do hospital.
Produz protocolos, laudos, perguntas frequentes e registros de pacientes.
"""

import json
import random
import uuid
from pathlib import Path
from faker import Faker

fake = Faker("pt_BR")

DIAGNOSES = [
    "Hipertensão arterial sistêmica",
    "Diabetes mellitus tipo 2",
    "Infarto agudo do miocárdio",
    "Acidente vascular cerebral isquêmico",
    "Pneumonia bacteriana",
    "Insuficiência cardíaca congestiva",
    "Doença renal crônica",
    "Hipotireoidismo",
    "Anemia ferropriva",
    "DPOC",
]

MEDICATIONS = [
    "Metformina 850mg",
    "Losartana 50mg",
    "AAS 100mg",
    "Atorvastatina 20mg",
    "Omeprazol 20mg",
    "Levotiroxina 50mcg",
    "Furosemida 40mg",
    "Espironolactona 25mg",
    "Amoxicilina 500mg",
    "Prednisona 20mg",
]

EXAMS = [
    "Hemograma completo",
    "Glicemia em jejum",
    "HbA1c",
    "Creatinina sérica",
    "TSH",
    "T4 livre",
    "Colesterol total e frações",
    "Triglicerídeos",
    "TGO/TGP",
    "Urina tipo I",
    "ECG 12 derivações",
    "Ecocardiograma",
    "Raio-X de tórax",
]

PROTOCOLS = [
    {
        "title": "Protocolo de Atendimento ao Paciente com IAM",
        "content": (
            "1. Realizar ECG em até 10 minutos da chegada.\n"
            "2. Administrar AAS 300mg mastigado imediatamente.\n"
            "3. Acionar equipe de hemodinâmica se supradesnivelamento de ST.\n"
            "4. Monitorização contínua de sinais vitais e saturação.\n"
            "5. Acesso venoso periférico calibroso.\n"
            "6. Coleta de troponina, CK-MB e BNP.\n"
            "7. Oxigênio suplementar se SatO2 < 94%.\n"
            "8. Heparina não-fracionada conforme peso do paciente."
        ),
    },
    {
        "title": "Protocolo de Controle Glicêmico Hospitalar",
        "content": (
            "Meta glicêmica: 140-180 mg/dL para pacientes internados em UTI.\n"
            "Meta glicêmica: 100-180 mg/dL para pacientes em enfermaria.\n"
            "Monitorar glicemia capilar a cada 4-6 horas.\n"
            "Hipoglicemia (<70 mg/dL): administrar 20g de glicose IV ou VO.\n"
            "Hiperglicemia (>250 mg/dL): avaliar cetonemia e acidose.\n"
            "Ajuste de insulina conforme escala dinâmica aprovada pela comissão."
        ),
    },
    {
        "title": "Protocolo de Sepse — Hora Zero",
        "content": (
            "Bundle da primeira hora (Sepse-3):\n"
            "1. Dosar lactato sérico.\n"
            "2. Hemoculturas (2 pares) antes de antibiótico.\n"
            "3. Antibiótico de amplo espectro em até 1 hora.\n"
            "4. Reposição volêmica: 30 mL/kg de cristaloide em hipotensão ou lactato ≥4.\n"
            "5. Vasopressor se PAM < 65 mmHg após volume.\n"
            "6. Reavaliação contínua da perfusão tecidual."
        ),
    },
    {
        "title": "Protocolo de Prevenção de TVP e TEP",
        "content": (
            "Estratificação de risco com escore de Caprini obrigatória na admissão.\n"
            "Baixo risco: deambulação precoce e meias de compressão graduada.\n"
            "Risco moderado/alto: heparina de baixo peso molecular (enoxaparina).\n"
            "Contraindicações: sangramento ativo, trombocitopenia < 50.000.\n"
            "Monitorar contagem plaquetária a cada 48h em uso de heparina não-fracionada."
        ),
    },
    {
        "title": "Protocolo de Prescrição de Antimicrobianos",
        "content": (
            "Toda prescrição de antimicrobiano deve ter: indicação clínica, agente suspeito, duração prevista.\n"
            "Coleta de culturas antes do início sempre que possível.\n"
            "Reavaliação obrigatória em 48-72h (de-escalação se possível).\n"
            "Carbapenêmicos requerem autorização da CCIH.\n"
            "Registrar alergias confirmadas (mecanismo e reação) no prontuário."
        ),
    },
]

FAQ_QA = [
    {
        "question": "Qual é a dose de AAS para prevenção secundária cardiovascular?",
        "answer": (
            "A dose recomendada de AAS para prevenção secundária cardiovascular é de 75 a 100 mg/dia, "
            "administrada por via oral, preferencialmente com refeições para reduzir irritação gástrica. "
            "Fonte: Protocolo Interno — Cardiologia, revisão 2024."
        ),
    },
    {
        "question": "Quando indicar hemodiálise de urgência?",
        "answer": (
            "As indicações absolutas de hemodiálise de urgência (mnemônico AEIOU): "
            "Acidose metabólica grave refratária, Eletrólitos (hipercalemia > 6,5 mEq/L refratária), "
            "Intoxicações (álcool, lítio, salicilatos), Overload (sobrecarga volêmica refratária), "
            "Uremia sintomática (encefalopatia, pericardite). "
            "Fonte: Protocolo de Nefrologia — UTI Adulto."
        ),
    },
    {
        "question": "Qual o critério para transferência de paciente à UTI?",
        "answer": (
            "Critérios de admissão em UTI incluem: instabilidade hemodinâmica necessitando vasopressores, "
            "insuficiência respiratória aguda com necessidade de ventilação mecânica, "
            "rebaixamento do nível de consciência com risco de via aérea, "
            "distúrbios metabólicos graves não responsivos ao tratamento inicial, "
            "monitorização invasiva necessária. Decisão final é do intensivista de plantão. "
            "Fonte: Regulação Interna de Leitos — Protocolo Hospitalar."
        ),
    },
    {
        "question": "Como tratar hipocalemia grave no paciente internado?",
        "answer": (
            "Hipocalemia grave (K < 3,0 mEq/L): reposição IV com KCl 10 mEq/hora em veia periférica "
            "(máx 20 mEq/hora em veia central com monitorização contínua). "
            "Verificar magnésio sérico — hipomagnesemia dificulta a correção do potássio. "
            "Reavaliação laboratorial após 2-4h de reposição. "
            "Via oral preferível se K entre 3,0-3,5 mEq/L e paciente tolerando. "
            "Fonte: Protocolo de Distúrbios Hidroeletrolíticos."
        ),
    },
    {
        "question": "Quando solicitar troponina e qual o seu valor de corte?",
        "answer": (
            "A troponina de alta sensibilidade (hs-TnI ou hs-TnT) deve ser solicitada em toda suspeita "
            "de síndrome coronariana aguda. O valor de corte varia conforme o ensaio utilizado no laboratório "
            "do hospital (percentil 99 da população de referência). Coleta na admissão e após 1-3 horas "
            "conforme protocolo de regra rápida de inclusão/exclusão. "
            "Elevações podem ocorrer em outras condições: miocardite, TEP, sepse, ICC descompensada. "
            "Fonte: Protocolo de Dor Torácica — Pronto-Socorro."
        ),
    },
    {
        "question": "Qual é o protocolo de manejo da anafilaxia?",
        "answer": (
            "Anafilaxia: emergência médica — agir imediatamente.\n"
            "1. Adrenalina 0,3-0,5 mg IM na face anterolateral da coxa (1:1000). Pode repetir a cada 5-15 min.\n"
            "2. Posição supina com MMII elevados (exceto vômito/dispneia).\n"
            "3. Oxigênio suplementar.\n"
            "4. Acesso venoso + reposição volêmica se hipotensão.\n"
            "5. Anti-histamínico (difenidramina) e corticoide como terapia adjuvante — NÃO substituem adrenalina.\n"
            "6. Observação por pelo menos 4-6h após estabilização.\n"
            "Fonte: Protocolo de Emergências Clínicas."
        ),
    },
]

LAUDO_TEMPLATES = [
    {
        "type": "ECG",
        "template": (
            "Ritmo sinusal. FC {fc} bpm. Eixo elétrico normal. "
            "Intervalo PR {pr} ms. Duração QRS {qrs} ms. "
            "Intervalo QTc {qtc} ms. {finding}. "
            "Conclusão: {conclusion}."
        ),
        "findings": [
            (
                "Sem alterações de repolarização ventricular",
                "ECG dentro dos limites da normalidade",
            ),
            (
                "Supradesnivelamento de ST em V1-V4",
                "Padrão compatível com lesão subepicárdica anterior — correlacionar clinicamente",
            ),
            (
                "Infradesnivelamento de ST em DII, DIII e aVF",
                "Alterações sugestivas de isquemia subendocárdica inferior",
            ),
            (
                "Bloqueio de ramo esquerdo completo",
                "BRE completo — dificulta análise de isquemia, correlacionar com clínica",
            ),
        ],
    },
    {
        "type": "Raio-X de Tórax",
        "template": (
            "Técnica: PA em inspiração. {finding}. "
            "Seios costofrênicos {cfr}. Área cardíaca {cardiac}. "
            "Conclusão: {conclusion}."
        ),
        "findings": [
            (
                "Transparência pulmonar preservada bilateralmente. Trama vásculo-brônquica normal",
                "livres",
                "normal (ICT < 0,5)",
                "Radiografia de tórax sem alterações significativas",
            ),
            (
                "Hipotransparência em base direita com velamento do seio costofrênico direito",
                "direito velado",
                "normal",
                "Padrão compatível com derrame pleural direito — avaliar etiologia",
            ),
            (
                "Infiltrado intersticial difuso bilateral com padrão em vidro fosco",
                "livres",
                "normal",
                "Padrão compatível com pneumonite — diagnóstico diferencial inclui COVID-19, PCP, edema pulmonar atípico",
            ),
        ],
    },
]


def generate_patient_record() -> dict:
    age = random.randint(18, 85)
    diagnosis = random.choice(DIAGNOSES)
    meds = random.sample(MEDICATIONS, k=random.randint(1, 4))
    pending_exams = random.sample(EXAMS, k=random.randint(1, 5))

    return {
        "patient_id": str(uuid.uuid4())[:8].upper(),
        "name": "[ANONIMIZADO]",
        "age": age,
        "sex": random.choice(["M", "F"]),
        "diagnosis": diagnosis,
        "medications": meds,
        "pending_exams": pending_exams,
        "admission_date": fake.date_this_year().isoformat(),
        "ward": random.choice(["UTI Adulto", "Enfermaria Clínica", "Pronto-Socorro", "Cardiologia"]),
        "alerts": _generate_alerts(diagnosis),
    }


def _generate_alerts(diagnosis: str) -> list[str]:
    alerts = []
    if "Diabetes" in diagnosis:
        alerts.append("Monitorar glicemia capilar a cada 6 horas")
    if "Hipertensão" in diagnosis:
        alerts.append("Verificar PA bilateral na admissão")
    if "Cardíaca" in diagnosis or "Miocárdio" in diagnosis:
        alerts.append("Monitorização cardíaca contínua")
    if "Renal" in diagnosis:
        alerts.append("Ajuste de doses de medicamentos para TFG")
    return alerts


def generate_ecg_laudo() -> dict:
    template_data = LAUDO_TEMPLATES[0]
    finding_data = random.choice(template_data["findings"])
    text = template_data["template"].format(
        fc=random.randint(55, 110),
        pr=random.randint(120, 200),
        qrs=random.randint(70, 120),
        qtc=random.randint(380, 480),
        finding=finding_data[0],
        conclusion=finding_data[1],
    )
    return {"type": "ECG", "text": text}


def generate_xray_laudo() -> dict:
    template_data = LAUDO_TEMPLATES[1]
    finding_data = random.choice(template_data["findings"])
    text = template_data["template"].format(
        finding=finding_data[0],
        cfr=finding_data[1],
        cardiac=finding_data[2],
        conclusion=finding_data[3],
    )
    return {"type": "Raio-X de Tórax", "text": text}


def build_fine_tuning_dataset(n_patients: int = 50) -> list[dict]:
    """
    Constrói o dataset no formato instruction-tuning (instruction / input / output).
    Combina FAQ, protocolos e laudos em pares de treino.
    """
    samples = []

    # Pares de FAQ
    for qa in FAQ_QA:
        samples.append({
            "instruction": "Você é um assistente médico hospitalar. Responda com base nos protocolos internos.",
            "input": qa["question"],
            "output": qa["answer"],
        })

    # Pares de protocolo — resumo
    for proto in PROTOCOLS:
        samples.append({
            "instruction": "Resuma o protocolo médico a seguir em linguagem clara para a equipe.",
            "input": proto["title"],
            "output": proto["content"],
        })

    # Laudos sintéticos
    for _ in range(n_patients):
        laudo = random.choice([generate_ecg_laudo, generate_xray_laudo])()
        samples.append({
            "instruction": "Interprete o laudo médico a seguir e destaque os achados clinicamente relevantes.",
            "input": laudo["text"],
            "output": f"Laudo de {laudo['type']} analisado. {laudo['text'].split('Conclusão:')[-1].strip()}",
        })

    return samples


def save_synthetic_data(output_dir: str = "data/synthetic") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Protocolos
    with open(f"{output_dir}/protocols.json", "w", encoding="utf-8") as f:
        json.dump(PROTOCOLS, f, ensure_ascii=False, indent=2)

    # FAQ
    with open(f"{output_dir}/faq.json", "w", encoding="utf-8") as f:
        json.dump(FAQ_QA, f, ensure_ascii=False, indent=2)

    # Registros de pacientes
    patients = [generate_patient_record() for _ in range(100)]
    with open(f"{output_dir}/patients.json", "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=2)

    # Dataset de fine-tuning
    ft_dataset = build_fine_tuning_dataset(n_patients=50)
    with open(f"{output_dir}/fine_tuning_dataset.json", "w", encoding="utf-8") as f:
        json.dump(ft_dataset, f, ensure_ascii=False, indent=2)

    print(f"Dados sintéticos salvos em '{output_dir}/'")
    print(f"  - {len(PROTOCOLS)} protocolos")
    print(f"  - {len(FAQ_QA)} pares FAQ")
    print(f"  - 100 registros de pacientes (anonimizados)")
    print(f"  - {len(ft_dataset)} amostras de fine-tuning")


if __name__ == "__main__":
    save_synthetic_data()
