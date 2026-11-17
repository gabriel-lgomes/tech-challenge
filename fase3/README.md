# Tech Challenge — Fase 3: Assistente Médico Virtual com LLM

## Objetivo

Criar um **assistente virtual médico** treinado com dados internos do hospital, capaz de auxiliar nas condutas clínicas, responder dúvidas de médicos e sugerir procedimentos com base nos protocolos internos. O sistema utiliza **fine-tuning de LLM** (Phi-3-mini com QLoRA), **LangChain** para orquestração do pipeline RAG e **LangGraph** para fluxos de decisão automatizados.

## Arquitetura

```
Pergunta clínica
      │
      ▼
[Guardrails] ── bloqueio se indevida
      │
      ▼
[RAG — ChromaDB] ── recupera protocolos relevantes
      │
      ▼
[Contexto do paciente — SQLite] ── prontuário, exames, alertas
      │
      ▼
[LLM fine-tunada (Phi-3-mini / QLoRA)] ── gera resposta
      │
      ▼
[Disclaimer + Fontes + Logging JSONL]
      │
      ▼
   Resposta
```

### Componentes principais

1. **Fine-tuning (QLoRA)** — Phi-3-mini fine-tunado com dados médicos sintéticos (protocolos, laudos e FAQ) usando Unsloth + QLoRA. Reduz VRAM para ~6GB, compatível com Colab T4 gratuito.
2. **RAG (ChromaDB)** — Recuperação semântica dos protocolos hospitalares e FAQ para contextualizar cada resposta.
3. **Assistente (LangChain)** — Pipeline que integra RAG + contexto estruturado do paciente (SQLite) + LLM fine-tunada em uma cadeia coesa.
4. **Fluxo automatizado (LangGraph)** — Grafo de decisão que, ao receber um ID de paciente, executa automaticamente: verificar exames pendentes → sugerir tratamento → emitir alertas → gerar resumo.

## Estrutura do projeto

```
fase3/
├── data/
│   ├── raw/                        # Datasets externos (PubMedQA, MedQuAD)
│   ├── synthetic/                  # Dados sintéticos gerados automaticamente
│   └── processed/                  # Dataset formatado para fine-tuning (HF format)
├── notebooks/
│   └── 01_fine_tuning_colab.ipynb  # Fine-tuning completo no Google Colab
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py  # Gera protocolos, laudos, FAQ e prontuários sintéticos
│   │   └── preprocessing.py        # Anonimização, formatação Alpaca e split treino/val
│   ├── fine_tuning/
│   │   └── trainer.py              # Pipeline QLoRA (Unsloth + SFTTrainer)
│   ├── assistant/
│   │   ├── chains.py               # Pipeline LangChain principal (RAG + LLM)
│   │   ├── retriever.py            # RAG com ChromaDB e embeddings multilíngues
│   │   └── database.py             # Prontuários e registros em SQLite
│   ├── graph/
│   │   └── medical_graph.py        # Fluxos LangGraph (5 nós de decisão)
│   └── security/
│       ├── guardrails.py           # Bloqueio de prescrições diretas e validação
│       └── audit_logger.py         # Logging estruturado em JSONL
├── models/                         # Modelo fine-tunado (gerado localmente — não versionado)
├── logs/                           # Logs de auditoria (JSONL — não versionados)
├── main.py                         # CLI principal
├── requirements.txt
└── .env.example
```

## Datasets utilizados

| Dataset | Conteúdo | Acesso |
|---|---|---|
| **PubMedQA** | Perguntas e respostas clínicas baseadas em publicações médicas | pubmedqa.github.io |
| **MedQuAD** | Conjunto de Q&A sobre saúde (NLM/NIH) | github.com/abachaa/MedQuAD |
| **Dados sintéticos** | Protocolos, laudos (ECG/RX), FAQ e prontuários gerados por `synthetic_generator.py` | Local |

Os dados sintéticos simulam o ambiente hospitalar com 5 protocolos clínicos, 6 pares de FAQ, laudos de ECG e Raio-X e 100 registros de pacientes anonimizados.

## Como executar

### Pré-requisitos

- Python 3.10+
- GPU com 8GB+ VRAM para fine-tuning local **ou** Google Colab (gratuito — recomendado)
- Chave Groq gratuita em [console.groq.com](https://console.groq.com) — permite rodar o assistente **sem GPU**

### Instalação

```bash
cd fase3
pip install -r requirements.txt
cp .env.example .env
# Edite .env com suas chaves (GROQ_API_KEY é suficiente para rodar sem GPU)
```

### 1. Setup — gera dados e constrói o índice

```bash
python main.py setup
```

Isso executa em sequência: geração de dados sintéticos → preprocessing → criação do banco SQLite → indexação no ChromaDB.

### 2. Fine-tuning (Google Colab recomendado)

Abra `notebooks/01_fine_tuning_colab.ipynb` no Google Colab (Runtime → Change runtime type → T4 GPU) e execute todas as células.

Ou localmente (requer GPU com ≥8GB VRAM):

```bash
python main.py finetune --model unsloth/Phi-3-mini-4k-instruct --epochs 3
```

### 3. Assistente interativo

```bash
python main.py ask
# O sistema pedirá o ID do paciente (opcional) e abrirá o modo de perguntas
```

### 4. Fluxo automatizado com LangGraph

```bash
python main.py flow <PATIENT_ID>
# Exemplo:
python main.py flow A1B2C3D4
```

O fluxo executa automaticamente os 5 nós e imprime o resumo completo com exames pendentes, sugestão de conduta e alertas emitidos.

## Fluxo LangGraph

```
[receber_paciente]
        │
        ▼
[verificar_exames] ── consulta SQLite + assistente LangChain
        │
        ▼
[sugerir_tratamento] ── assistente LangChain + protocolos internos
        │
        ▼
[emitir_alertas] ── grava alertas no banco e no log de auditoria
        │
        ▼
[finalizar] ── monta resumo completo
        │
        ▼
       END
```

## Segurança e validação

| Mecanismo | Implementação |
|---|---|
| **Guardrails** | `src/security/guardrails.py` — bloqueia perguntas que tentam obter prescrições diretas ou diagnósticos definitivos |
| **Logging de auditoria** | `logs/assistant.jsonl` — cada interação salva com timestamp, sessão, fontes e flags de risco |
| **Explainability** | Toda resposta lista as fontes consultadas (protocolo ou FAQ específico) |
| **Disclaimer automático** | Todas as respostas incluem aviso de que decisões clínicas requerem validação médica |

## Modelos suportados

| Modo | Modelo | Requisito |
|---|---|---|
| Fine-tuned local | Phi-3-mini (QLoRA, 4-bit) | GPU ≥8GB VRAM |
| Fine-tuned Colab | Phi-3-mini (QLoRA, 4-bit) | Google Colab T4 (gratuito) |
| API sem GPU | LLaMA 3.1 8B via Groq | Chave API gratuita |

O sistema detecta automaticamente se o modelo fine-tunado está disponível em `models/fine_tuned/`; caso contrário, usa a API Groq como fallback.

## Continuidade com as fases anteriores

| Fase | Foco | Dataset |
|---|---|---|
| Fase 1 | Classificação de câncer de mama (ML + SHAP) | Breast Cancer Wisconsin |
| Fase 2 | Otimização genética + laudos com LLM via API | Breast Cancer Wisconsin |
| **Fase 3** | **Fine-tuning de LLM + assistente médico + LangGraph** | **PubMedQA + MedQuAD + Sintético** |
