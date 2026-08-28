# Tech Challenge — IA para Saúde Feminina

Sistema inteligente de suporte ao diagnóstico médico com foco na saúde e segurança da mulher, utilizando Machine Learning, Algoritmo Genético e LLM para identificação precoce de câncer de mama.

## Objetivo

Construir modelos preditivos de classificação sobre dados médicos relacionados à saúde feminina, aplicando técnicas de Machine Learning, análise exploratória, explicabilidade, otimização evolutiva de hiperparâmetros e interpretabilidade em linguagem natural via LLM.

## Datasets

### Tarefa principal — dados estruturados
**Breast Cancer Wisconsin Diagnostic Dataset**
Classificação de tumores de mama como malignos ou benignos a partir de características extraídas de biópsias (FNA).

- Fonte: [Kaggle — UCI ML Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- 569 amostras | 30 features | Variável alvo: `diagnosis` (M = maligno, B = benigno)

### Extra — visão computacional
**Breast Ultrasound Images Dataset (BUSI)**
Classificação de ultrassonografias de mama em três categorias: benigno, maligno e normal.

- Fonte: [Kaggle — Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- 780 imagens | 3 classes: benigno (437), maligno (210), normal (133)

## Estrutura do projeto

```
tech-challenge/
├── data/
│   ├── raw/
│   │   ├── data.csv                       # Dataset Wisconsin (estruturado)
│   │   └── images/Dataset_BUSI_with_GT/   # Imagens BUSI (baixar via Kaggle)
│   └── processed/                         # Dados pré-processados (gerados localmente)
├── notebooks/
│   ├── 01_eda.ipynb                  # Análise exploratória de dados
│   ├── 02_preprocessing.ipynb        # Pré-processamento e pipeline
│   ├── 03_modeling.ipynb             # Modelagem ML, avaliação e SHAP
│   ├── 04_cnn.ipynb                  # [EXTRA] CNN com Transfer Learning (MobileNetV2)
│   ├── 05_genetic_algorithm.ipynb    # [FASE 2] Otimização com Algoritmo Genético (DEAP)
│   └── 06_llm_interpretability.ipynb # [FASE 2] Interpretabilidade com LLM via Groq API
├── src/
│   ├── preprocessing.py    # Funções reutilizáveis de pré-processamento
│   └── evaluation.py       # Funções de avaliação e métricas
├── models/                 # Modelos treinados (gerados localmente)
├── reports/
│   └── figures/            # Gráficos e visualizações geradas (01–24)
├── requirements.txt
└── README.md
```

## Modelos utilizados

| Notebook | Modelo | Descrição |
|---|---|---|
| `03_modeling` | **Regressão Logística** | Baseline linear, rápido e interpretável |
| `03_modeling` | **Random Forest** | Ensemble de árvores, captura relações não-lineares |
| `04_cnn` (extra) | **CNN MobileNetV2** | Transfer Learning para classificação de ultrassonografias |
| `05_genetic_algorithm` | **Random Forest (otimizado)** | Hiperparâmetros encontrados via Algoritmo Genético |
| `06_llm_interpretability` | **Qwen 3 27B (Groq)** | Geração de laudos clínicos a partir de valores SHAP |

## Resultados

| Modelo | Accuracy | Recall | Precision | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Regressão Logística | — | — | — | — | — |
| Random Forest (original) | 0.9474 | 0.881 | 0.9737 | 0.925 | 0.994 |
| Random Forest (AG) | 0.9474 | 0.881 | 0.9737 | 0.925 | 0.990 |

> Regressão Logística: valores gerados ao executar `03_modeling.ipynb`. Random Forest: resultados do conjunto de teste (20%).

## Métricas de avaliação

Dado o contexto médico, o **Recall** é a métrica mais crítica — um falso negativo (não detectar um tumor maligno) tem consequências graves para a paciente. São reportadas também accuracy, precision, F1-score e AUC-ROC.

## Explicabilidade

- **Feature Importance** (Random Forest): importância global de cada variável
- **SHAP Beeswarm**: impacto de cada feature em cada amostra individualmente
- **SHAP Waterfall**: explicação detalhada de um caso específico
- **LLM (Groq API)**: laudos clínicos em português gerados a partir dos valores SHAP

## Como executar

### Pré-requisitos

- Python 3.10+ (recomendado via [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Chave de API do Groq (gratuita em [console.groq.com](https://console.groq.com)) — necessária apenas para o notebook 06

### Instalação

```bash
pip install -r requirements.txt
```

### Download dos datasets

```bash
pip install kaggle

# Dataset Wisconsin (dados estruturados)
kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p data/raw --unzip

# Dataset BUSI (imagens para CNN — extra)
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p data/raw/images --unzip
```

### Configuração da API Groq (notebook 06)

Crie um arquivo `.env` na raiz do projeto com:

```
GROQ_API_KEY=sua_chave_aqui
```

### Execução

Execute os notebooks na ordem dentro de `notebooks/`:

```
01_eda.ipynb                → Análise exploratória
02_preprocessing.ipynb      → Limpeza, encoding e pipeline
03_modeling.ipynb           → Treinamento, avaliação e SHAP
04_cnn.ipynb                → [EXTRA] CNN com Transfer Learning (requer TensorFlow)
05_genetic_algorithm.ipynb  → [FASE 2] Otimização de hiperparâmetros com AG
06_llm_interpretability.ipynb → [FASE 2] Laudos clínicos com LLM (requer .env)
```

## Discussão crítica

Os modelos têm caráter de **apoio à decisão**, não de diagnóstico autônomo. O médico responsável deve sempre ter a palavra final, utilizando as predições como ferramenta auxiliar de triagem.

Limitações:
- O dataset Wisconsin é coletado em condições controladas; performance em dados reais pode variar
- Qualquer falso negativo representa risco clínico grave
- O modelo deve ser apresentado como "score de risco", não como diagnóstico definitivo
