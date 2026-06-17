# Consolidação de Entrega: Predição de Atrasos no E-Commerce

Este diretório contém todos os artefatos necessários para a reprodução completa do estudo **"PREDIÇÃO DE ATRASOS NA ENTREGA DE PEDIDOS EM E-COMMERCE: UM ESTUDO COMPARATIVO ENTRE ALGORITMOS DE MACHINE LEARNING"**, com base no dataset público da Olist.

## Conteúdo da Pasta

- **pipeline_artigo.ipynb** — Notebook mestre. Contém todo o ciclo CRISP-DM: carregamento, EDA, engenharia de atributos (incluindo Haversine), modelagem, avaliação comparativa e persistência de artefatos.
- **Artigo - Predição de atrasos.pdf** — Versão final do artigo de referência.
- **RELATORIO_PREDICAO_ATRASOS.md** — Relatório completo, com Sumário Executivo no topo e detalhamento CRISP-DM em 8 seções.
- **requirements.txt** — Dependências Python.
- **datasets/** — Datasets brutos da Olist (9 arquivos CSV).
- **reports/**
  - `comparacao_modelos.csv` — métricas dos 3 modelos obtidas neste experimento.
  - `comparacao_obtido_vs_artigo.csv` — comparação lado a lado com a Tabela 4 do artigo.
  - `figures/` — gráficos gerados pelo notebook (`eda_target_distribution.png`, `eda_features.png`, `comparacao_obtido_vs_artigo.png`).
- **models/pipeline_atraso.pkl** — Pipeline serializado do melhor modelo (HistGradientBoosting + pré-processamento + SMOTE), selecionado por AUC-ROC.

## Como Reproduzir os Resultados

Pré-requisito: Python 3.10+ (validado em 3.14.5).

```bash
# 1. Criar ambiente virtual
python -m venv .venv

# 2. Ativar
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate              # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar o notebook (a partir desta pasta, para que o caminho relativo ./datasets/ resolva)
jupyter nbconvert --to notebook --execute --inplace pipeline_artigo.ipynb
# ou abrir no Jupyter Lab/Notebook e rodar célula a célula
```

O notebook regenera todos os artefatos em `reports/` e `models/` a partir do zero.

## Verificação de Integridade (Valores Obtidos)

Métricas reais produzidas pelo notebook nesta execução (split temporal 80/20, SMOTE com `sampling_strategy=0.3`, `random_state=42`):

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| Regressão Logística | 0,6810 | 0,0447 | 0,2468 | 0,0757 | 0,4473 |
| Random Forest | 0,9453 | 0,0952 | 0,0039 | 0,0075 | 0,5321 |
| **HistGradientBoosting** | **0,9470** | **0,3333** | 0,0010 | 0,0020 | **0,5739** |

**Modelo selecionado (`pipeline_atraso.pkl`):** HistGradientBoosting — critério: maior AUC-ROC (0,5739).

Para comparação completa com os valores do artigo de referência, ver `reports/comparacao_obtido_vs_artigo.csv` ou a seção 6 do `RELATORIO_PREDICAO_ATRASOS.md`.

---
**Status:** Entrega consolidada. Pendente: rodada de refinamento (threshold tuning, hyperparameters).
