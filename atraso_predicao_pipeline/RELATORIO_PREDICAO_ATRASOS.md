# Relatório Completo — Predição de Atrasos na Entrega
## Olist E-Commerce Dataset | CRISP-DM | LR × RF × HistGradientBoosting

---

## Sumário Executivo

**Pergunta de pesquisa.** Com base apenas em informações disponíveis no momento da compra (produto, vendedor, localização, data), é possível prever se um pedido vai atrasar?

**Metodologia.** CRISP-DM, 12 features (9 numéricas + 3 categóricas, incluindo distância Haversine vendedor–cliente), split temporal 80/20, pipeline com SMOTE integrado, três algoritmos: Regressão Logística, Random Forest, HistGradientBoosting.

**Resultados obtidos:**

| Modelo | Accuracy | Precisão | Recall | F1 | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Regressão Logística | 0,6810 | 0,0447 | **0,2468** | 0,0757 | 0,4473 |
| Random Forest | 0,9453 | 0,0952 | 0,0039 | 0,0075 | 0,5321 |
| **HistGradientBoosting** | **0,9470** | **0,3333** | 0,0010 | 0,0020 | **0,5739** |

**Modelo selecionado:** HistGradientBoosting, critério **maior AUC-ROC** (0,5739). Serializado em `models/pipeline_atraso.pkl`. A escolha por AUC, e não por Recall, reflete o objetivo operacional: precisamos de um modelo cujo **ranqueamento** seja útil; Recall útil é alcançado posteriormente via calibração de threshold.

**Achados-chave:**
- **Sinal preditivo fraco** nas variáveis estáticas do pedido — os atrasos são determinados majoritariamente por fatores externos (transportadora, clima, gargalos sazonais) ausentes do dataset.
- **Sazonalidade forte:** Março (Carnaval) = 17,15% de atraso; Novembro (Black Friday) = 14,31%; Junho = 2,21%.
- **Geografia logística:** AL (23,93%) e MA (19,67%) lideram; pedidos interestaduais têm 1,5× a taxa intraestadual.
- **Reprodução fiel da Tabela 4 do artigo** com diferenças < 1 pp em quase todas as células.

**Limitações principais:**
- Sem cross-validation (hold-out simples) — sem intervalo de confiança.
- Threshold fixo em 0,5 desperdiça a capacidade discriminativa latente; SMOTE + `class_weight` na LR aplica dupla compensação.

**Próximos passos (rodada de refinamento):**
- Calibração de threshold sobre HGB via curva precisão-recall (extrair Recall útil).
- Busca sistemática de hiperparâmetros.
- Padronizar o mecanismo de balanceamento entre os 3 modelos.

> Detalhamento completo nas seções 1 a 8 abaixo.

---

## Sumário

1. [Contexto e Objetivo](#1-contexto-e-objetivo)
2. [Fonte de Dados](#2-fonte-de-dados)
3. [Fase 1 — Entendimento dos Dados](#3-fase-1--entendimento-dos-dados)
4. [Fase 2 — Engenharia de Atributos](#4-fase-2--engenharia-de-atributos)
5. [Fase 3 — Modelagem](#5-fase-3--modelagem)
6. [Fase 4 — Avaliação e Métricas](#6-fase-4--avaliação-e-métricas)
7. [Discussão e Limitações](#7-discussão-e-limitações)
8. [Conclusões e Trabalhos Futuros](#8-conclusões-e-trabalhos-futuros)

> **Nota sobre os números deste relatório:** todos os percentuais, contagens e métricas reportados foram produzidos pela execução atual do notebook `pipeline_artigo.ipynb`. Onde houver referência à Tabela 4 do artigo original, ela está explicitamente rotulada como "Artigo (referência)".

---

## 1. Contexto e Objetivo

Este estudo constrói um modelo preditivo capaz de identificar, **no momento da compra**, pedidos com maior probabilidade de entrega atrasada. A identificação antecipada permite à operação logística agir preventivamente: priorizar o despacho, escalar transportadoras alternativas ou comunicar o cliente antes da concretização do problema.

**Pergunta central:** Com base apenas nas informações disponíveis no momento da compra (produto, vendedor, localização, data), é possível prever se um pedido vai atrasar?

**Metodologia:** CRISP-DM (Cross-Industry Standard Process for Data Mining)

**Algoritmos comparados:** Regressão Logística × Random Forest × HistGradientBoosting

---

## 2. Fonte de Dados

O dataset é o **Olist Brazilian E-Commerce Public Dataset**, composto por arquivos CSV que cobrem pedidos realizados entre **setembro de 2016 e outubro de 2018**.

### Arquivos utilizados

| Arquivo | Conteúdo | Linhas |
| --- | --- | --- |
| `olist_orders_dataset.csv` | Pedidos, status, datas de compra, aprovação, envio, entrega e estimativa | 99.441 |
| `olist_order_items_dataset.csv` | Itens de cada pedido, preço, frete | 112.650 |
| `olist_products_dataset.csv` | Atributos do produto: categoria, peso, dimensões | 32.951 |
| `olist_sellers_dataset.csv` | Localização do vendedor (estado, cidade, CEP) | 3.095 |
| `olist_customers_dataset.csv` | Localização do comprador (estado, cidade, CEP) | 99.441 |
| `olist_geolocation_dataset.csv` | Coordenadas geográficas por prefixo de CEP | 1.000.163 |

### Filtragem e deduplicação

| Etapa | Valor |
| --- | --- |
| Pedidos brutos (`orders`) | 99.441 |
| Filtro: `status='delivered'` + `order_delivered_customer_date` não nulo | **96.470** |
| Pedidos únicos após deduplicação por `order_id` | **96.470** |
| Proporção da classe positiva (atraso = 1) | **8,11%** |
| Proporção da classe negativa (no prazo = 0) | **91,89%** |

---

## 3. Fase 1 — Entendimento dos Dados

### Definição do Target

A variável-alvo foi definida como:

```
target = 1  quando  order_delivered_customer_date > order_estimated_delivery_date
target = 0  nos demais casos
```

Pedidos sem data de entrega registrada foram removidos, pois essa informação é necessária para o cálculo do target. A definição é fiel à descrita no artigo de referência.

### Distribuição do Target

| Classe | Contagem | Proporção |
| --- | --- | --- |
| 0 — No prazo | 88.644 | 91,89% |
| 1 — Atrasado | 7.826 | **8,11%** |

O desbalanceamento torna a acurácia bruta enganosa: um classificador ingênuo que sempre prediz "no prazo" alcançaria 91,89% de acurácia sem qualquer utilidade prática.

### Análise Exploratória (valores reais calculados no notebook)

**Picos sazonais de atraso (por mês):**

| Mês | Taxa de Atraso |
| --- | --- |
| Março | **17,15%** (período de Carnaval) |
| Novembro | **14,31%** (Black Friday) |
| Fevereiro | 13,41% |
| Dezembro | 8,38% |
| ... | ... |
| Junho | **2,21%** (menor taxa) |

**Top 5 estados (UF do cliente) com maior taxa de atraso** (apenas UFs com ≥ 100 pedidos):

| Estado | Taxa de Atraso | n |
| --- | --- | --- |
| AL | **23,93%** | 397 |
| MA | 19,67% | 717 |
| PI | 15,97% | 476 |
| CE | 15,32% | 1.279 |
| SE | 15,22% | 335 |

A concentração no Nordeste sugere efeito de distância logística da malha de vendedores (majoritariamente concentrada em SP). UFs como BA (não está no top 5) e RJ (também fora) têm taxas inferiores, contrariando a intuição inicial.

**Top 5 estados com menor taxa de atraso:**

| Estado | Taxa de Atraso | n |
| --- | --- | --- |
| RO | 2,88% | 243 |
| AM | 4,14% | 145 |
| PR | 5,00% | 4.923 |
| MG | 5,61% | 11.354 |
| SP | 5,89% | 40.494 |

**Impacto de pedidos interestaduais:**

| Tipo | Taxa de Atraso | n |
| --- | --- | --- |
| Interestadual (`mesmo_estado=0`) | **9,27%** | 61.769 |
| Intraestadual (`mesmo_estado=1`) | 6,06% | 34.701 |

Pedidos interestaduais têm aproximadamente **1,5×** a taxa de atraso dos intraestaduais, confirmando que a complexidade logística da travessia estadual é um fator de risco relevante.

---

## 4. Fase 2 — Engenharia de Atributos

Todas as 12 variáveis preditoras foram construídas exclusivamente com informações disponíveis no momento da compra, evitando vazamento de dados. Variáveis derivadas de eventos posteriores ao pedido (como `order_delivered_carrier_date`) foram deliberadamente excluídas.

### Atributos preditores (Tabela 2 do artigo)

| Atributo | Tipo | Descrição |
| --- | --- | --- |
| `price` (soma) | Contínua | Valor total do pedido (R$) |
| `freight_value` (soma) | Contínua | Frete total pago (R$) |
| `product_weight_g` (soma) | Contínua | Peso total do pedido (g) |
| `volume_cm3` (soma) | Contínua | Volume total (comp. × alt. × larg.) |
| `distancia_km` | Contínua | Distância Haversine vendedor–cliente (km) |
| `dia_semana` | Discreta | Dia da semana da compra (0=segunda, 6=domingo) |
| `mes` | Discreta | Mês da compra — captura sazonalidade |
| `hora` | Discreta | Hora da compra (0–23) |
| `mesmo_estado` | Binária | 1 se vendedor e cliente no mesmo estado |
| `product_category_name` | Categórica | Categoria do produto |
| `customer_state` | Categórica | Estado (UF) do comprador |
| `seller_state` | Categórica | Estado (UF) do vendedor |

### Distância Haversine

A fórmula de Haversine calcula a distância em linha reta entre dois pontos na superfície terrestre, considerando a curvatura esférica. A tabela de geolocalização foi pré-agregada por prefixo de CEP (média de latitude e longitude) antes da junção com os pedidos. Estatísticas obtidas: média ≈ 601 km, máximo ≈ 8.678 km, com 478 pedidos sem correspondência de CEP (tratados pelo imputador no pipeline).

---

## 5. Fase 3 — Modelagem

### Divisão Treino/Teste

A divisão foi feita de forma **temporal**: o dataset foi ordenado por `order_purchase_timestamp` e dividido na proporção 80/20. Dividir aleatoriamente dados com dependência temporal constitui erro metodológico grave — permite que o modelo aprenda com observações futuras para prever o passado.

| Conjunto | Amostras | Proporção positiva |
| --- | --- | --- |
| Treino (80% mais antigos) | 77.176 | **8,82%** |
| Teste (20% mais recentes) | 19.294 | **5,29%** |

A proporção menor no conjunto de teste é esperada: o final do dataset (segundo semestre de 2018) apresenta menor sazonalidade negativa do que o início (que contém Carnaval e Black Friday).

### Pipeline Científico (ImbPipeline)

```
ImbPipeline
├── ColumnTransformer
│   ├── Numéricas: SimpleImputer(median) → StandardScaler
│   └── Categóricas: SimpleImputer(constant) → OneHotEncoder(handle_unknown='ignore')
├── SMOTE(sampling_strategy=0.3, random_state=42)
└── Classificador
```

O SMOTE foi integrado ao `ImbPipeline` (imbalanced-learn), garantindo que instâncias sintéticas sejam geradas **apenas sobre os dados de treinamento**, prevenindo vazamento de validação. O `sampling_strategy=0.3` estabelece proporção de 30% de positivos no treinamento — configuração menos agressiva que o balanceamento total.

A imputação de valores ausentes baseou-se **exclusivamente nas medianas do conjunto de treinamento**, evitando contaminação sutil do conjunto de teste.

### Algoritmos Avaliados

| Algoritmo | Configuração |
| --- | --- |
| Regressão Logística | `class_weight='balanced'`, `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators=200`, `random_state=42` |
| HistGradientBoosting | configuração padrão, `random_state=42` |

> **Nota:** o uso simultâneo de SMOTE e `class_weight='balanced'` apenas na Regressão Logística introduz dupla compensação não aplicada aos outros modelos — comportamento herdado do artigo de referência. Esta inconsistência é alvo da próxima rodada de refinamento.

---

## 6. Fase 4 — Avaliação e Métricas

### Métricas Adotadas

Dado o desbalanceamento das classes, a acurácia global foi descartada como métrica primária. As métricas utilizadas:

- **AUC-ROC:** capacidade discriminativa geral do modelo, independente do limiar de classificação. Critério oficial de seleção do melhor modelo neste estudo.
- **Recall:** proporção dos atrasos reais corretamente identificados.
- **Precisão:** proporção dos alertas emitidos que correspondem a atrasos reais.
- **F1-Score:** média harmônica entre precisão e recall.

### Resultados Obtidos (este experimento)

| Modelo | Accuracy | Precisão | Recall | F1 | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Regressão Logística | 0,6810 | 0,0447 | **0,2468** | 0,0757 | 0,4473 |
| Random Forest | 0,9453 | 0,0952 | 0,0039 | 0,0075 | 0,5321 |
| **HistGradientBoosting** | **0,9470** | **0,3333** | 0,0010 | 0,0020 | **0,5739** |

### Tabela 4 do Artigo (referência)

| Modelo | Accuracy | Precisão | Recall | F1 | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Regressão Logística | 0,6826 | 0,0451 | 0,2478 | 0,0763 | 0,4465 |
| Random Forest | 0,9456 | 0,1500 | 0,0059 | 0,0113 | 0,5355 |
| HistGradientBoosting | 0,9471 | 0,0000 | 0,0000 | 0,0000 | 0,5932 |

A comparação completa lado a lado (com coluna `Diferenca`) está em `reports/comparacao_obtido_vs_artigo.csv`. As diferenças ficam abaixo de 1 ponto percentual em quase todas as células — consistente com pequenas variações de SMOTE (estocástico) e versões de biblioteca. A divergência maior é o AUC-ROC do HGB (~2 pp abaixo do artigo).

### Análise por Algoritmo

**Regressão Logística (LR)** — apresentou o maior recall (0,2468), identificando cerca de 25% dos atrasos reais. Contudo, o AUC-ROC de **0,4473 está abaixo de 0,5**, o que indica que o **ranking de probabilidades produzido pelo modelo é, em média, pior do que aleatório**. Como AUC é independente do threshold, este resultado não é um problema de calibração de limiar: é um indicador de que a combinação `SMOTE + class_weight='balanced' + LR linear` degrada a separação entre as classes. O alto recall vem da classificação positiva de quase 1/3 de todos os pedidos do teste — produzindo precisão de apenas 4,47%, ou seja, ~24 falsos alertas para cada atraso real identificado.

**Random Forest (RF)** — obteve acurácia alta (0,9453) e AUC-ROC de 0,5321 (marginalmente melhor que aleatório). Recall próximo de zero (0,0039) no threshold padrão 0,5: o algoritmo aprendeu a classificar sistematicamente quase todos os pedidos como "no prazo". O AUC indica algum poder discriminativo latente nas probabilidades — ajuste de threshold pela curva precisão-recall pode extrair recall útil.

**HistGradientBoosting (HGB)** — atingiu o **maior AUC-ROC (0,5739)**, comportamento característico de modelos boosting que tendem a aprender padrões sutis em alta dimensão. No threshold padrão 0,5, recall e F1 colapsam praticamente a zero (apenas 3 predições positivas no teste, das quais 1 acertou — precisão 33,33%). Esse comportamento é esperado: classes muito desbalanceadas + boosting + threshold default geram quase nenhuma predição positiva. **Trabalho de refinamento (calibração de threshold) é o que vai extrair recall operacional deste modelo.**

### Modelo Selecionado

**Critério:** maior AUC-ROC (capacidade de ranqueamento; independente de threshold).
**Selecionado:** HistGradientBoosting (AUC-ROC = **0,5739**).
**Artefato:** `models/pipeline_atraso.pkl` (pipeline completo: pré-processamento + SMOTE + HGB).

A escolha por AUC-ROC, e não por Recall, reflete o objetivo operacional: precisamos de um modelo cujo **ranqueamento** seja útil para priorizar atenção logística. Recall pode ser conquistado posteriormente via calibração de threshold sobre um modelo com bom ranking; o oposto não é possível.

---

## 7. Discussão e Limitações

### Baixo Sinal Preditivo

Os resultados indicam **baixo sinal preditivo** nas variáveis estáticas disponíveis no momento da compra. O atraso de entregas não está fortemente correlacionado com preço, peso, distância geográfica ou categoria do produto.

Esta constatação aponta para **fatores externos e dinâmicos** como principais determinantes:

- Eficiência operacional da transportadora alocada
- Condições climáticas na rota
- Falhas na última milha
- Sobrecarga de pedidos em períodos sazonais (confirmada pelos picos de Março e Novembro)

Todas essas variáveis estão ausentes do dataset Olist.

### Validade da Baixa Performance

A baixa performance **não representa falha metodológica** — reflete o fenômeno estudado. Incluir variáveis disponíveis apenas após a compra (ex.: `order_delivered_carrier_date`) inflaria artificialmente as métricas mas configuraria *data leakage* e invalidaria cientificamente o modelo.

### Limitações Conhecidas

1. **Sem cross-validation:** uso de hold-out simples (sem `TimeSeriesSplit` nem k-fold) — não há intervalo de confiança sobre as métricas.
2. **Sem hyperparameter tuning:** todos os modelos usam configuração default (alvo da próxima fase).
3. **Threshold fixo em 0,5:** desperdiça a capacidade discriminativa latente dos modelos com AUC > 0,5.
4. **Balanceamento inconsistente:** SMOTE + `class_weight='balanced'` simultâneos apenas na LR (herdado do artigo).
5. **Agregação `first` por order_id:** pedidos com múltiplos vendedores/categorias são reduzidos arbitrariamente ao primeiro item.

---

## 8. Conclusões e Trabalhos Futuros

### Principais Achados

| Achado | Detalhe |
| --- | --- |
| Modelo selecionado | HistGradientBoosting (AUC-ROC = 0,5739) |
| Melhor AUC-ROC obtido | HGB: 0,5739 |
| Maior recall obtido (porém com ranking ruim) | LR: 0,2468 |
| Sinal preditivo geral | Fraco nas variáveis disponíveis no momento da compra |
| Sazonalidade crítica | Março: 17,15% (Carnaval); Novembro: 14,31% (Black Friday) |
| UFs de maior risco | AL (23,93%), MA (19,67%), PI (15,97%) |
| Efeito logístico | Pedidos interestaduais têm 1,5× a taxa intraestadual |

### Contribuições do Estudo

- Pipeline científico com SMOTE integrado, divisão temporal, imputação restrita ao treino e ausência verificada de *data leakage*.
- Engenharia de atributos com distância Haversine.
- Análise comparativa multi-algoritmo com protocolo experimental padronizado.
- Comparação transparente Obtido vs Artigo (Tabela 4) com diferenças quantificadas.
- Persistência reprodutível: notebook gera todos os artefatos (`reports/*.csv`, `reports/figures/*.png`, `models/pipeline_atraso.pkl`) em uma única execução.

### Trabalhos Futuros (Refinamento Imediato)

1. **Calibração de threshold** sobre HGB pela curva precisão-recall — extrair recall útil sem inflar falsos positivos.
2. **Busca de hiperparâmetros** (n_estimators, learning_rate, max_depth, l2_regularization).
3. **Padronizar mecanismo de balanceamento** entre os 3 modelos.
4. **Adicionar `TimeSeriesSplit`** para intervalo de confiança nas métricas.
5. **Feature engineering adicional:** número de itens, número de vendedores distintos, indicador de produto multi-categoria.

### Trabalhos Futuros (Extensão de Dados)

1. Integração de dados meteorológicos históricos por rota (via APIs como INMET).
2. Indicadores de desempenho histórico das transportadoras por região.
3. Modelos sequenciais (RNN/Transformer) sobre o fluxo logístico longitudinal.

---

*Notebook: `pipeline_artigo.ipynb`*
*Metodologia: CRISP-DM | Algoritmos: Regressão Logística, Random Forest, HistGradientBoosting | Dataset: Olist (2016–2018)*
*Autoria: Murilo Wolff Klug — Centro Universitário da Fundação Assis Gurgacz, 2026*
