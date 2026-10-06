# Relatório de Resultados — Baseline E3: TF-IDF + LinearSVC

**Data:** 2026-10-01 13:10
**Arquitetura:** TF-IDF (n-gram 1-3, 20.000 features, l2) + `LinearSVC(C=0.1, random_state=42)`

## 1. Comparação das 3 Razões de Treino no Conjunto Dev In-Domain (ndmais)

| Configuração | Razão Treino | Dev AP (PR-AUC) | Dev F1 (Corte 0.0) | Limiar Dev Ótimo | Dev F1 (Limiar Dev) | Dev Recall | Dev Precision |
|---|---|---|---|---|---|---|---|
| `natural` | 1:106.4 | **0.6446** | 0.3288 | -0.593 | **0.6015** | 0.6780 | 0.5405 |
| `1a64` | 1:64.0 | **0.6309** | 0.4444 | -0.500 | **0.6165** | 0.6949 | 0.5541 |
| `1a10` | 1:10.0 | **0.5674** | 0.5000 | -0.182 | **0.5714** | 0.6780 | 0.4938 |

> **Configuração Vencedora Selecionada:** `natural` (maior AP no dev: **0.6446**). Limiar fixado: **-0.5935**.

## 2. Desempenho do Modelo Vencedor: In-Domain vs. Exposição Cross-Portal

| Cenário de Avaliação | Dataset | AP (PR-AUC) [95% CI] | F1 [95% CI] | Recall [95% CI] | Precision | % Previsto Fraude |
|---|---|---|---|---|---|---|
| **In-Domain (ndmais)** | `test.csv` (n=9091) | **0.7001** [0.5933 - 0.7961] | **0.6091** [0.5217 - 0.6816] | 0.7407 [0.6410 - 0.8333] | 0.5172 | 1.28% |
| **Cross-Portal (Exposição)** | `exposicao_1a106.csv` (n=11379) | **0.5065** [0.4080 - 0.6046] | **0.4663** [0.3684 - 0.5548] | 0.3585 [0.2685 - 0.4490] | 0.6667 | 0.50% |

## 3. Diagnóstico de Viés de Domínio (Shift de Limiar)

- **F1 com Limiar do Dev:** `0.4663`
- **F1 com Teto Ótimo na Exposição:** `0.5359` (corte reajustado para `-0.708`)
- **Interpretação:** O modelo sofreu degradação substancial de ranqueamento ao ser exposto a fontes desconhecidas.
