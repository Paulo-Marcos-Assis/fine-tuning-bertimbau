# Resultados — Baseline E3: TF-IDF + LinearSVC (ndmais only)

**Arquitetura:** TF-IDF (n-gram 1–3, 20k features, norm=l2) + LinearSVC(C=0.1)  
**Treino:** `train_natural.csv` — razão 1:106.4 (configuração vencedora no dev)  
**Limiar de decisão:** −0.5935 (calibrado no dev)

---

## Desempenho por Estágio

| Estágio | Dataset | N total | Positivos | F1 [95% CI] | Recall [95% CI] | Precision |
|---|---|---|---|---|---|---|
| Dev | dev.csv | 7.351 | 59 | 0.6015 | 67.80% | 54.05% |
| Teste In-Domain | test.csv | 9.091 | 81 | 0.6091 [0.52–0.68] | 74.07% [64–83%] | 51.72% |
| Cross-Portal | exposicao_1a106.csv | 11.379 | 106 | 0.4663 [0.37–0.55] | 35.85% [27–45%] | 66.67% |

---

## Top 30 Features — Peso para Classe FRAUDE

| # | Feature | Coeficiente LinearSVC |
|---|---|---|
| 1 | veigamed | +1.003 |
| 2 | mensageiro | +0.922 |
| 3 | empresa | +0.908 |
| 4 | licitação | +0.903 |
| 5 | operação | +0.814 |
| 6 | fraude | +0.692 |
| 7 | contratos | +0.597 |
| 8 | operação mensageiro | +0.589 |
| 9 | contrato | +0.538 |
| 10 | golpes | +0.522 |
| 11 | irregularidades | +0.498 |
| 12 | serrana | +0.497 |
| 13 | corrupção | +0.492 |
| 14 | fraudes | +0.469 |
| 15 | scpar | +0.455 |
| 16 | alfa | +0.440 |
| 17 | respiradores | +0.435 |
| 18 | propina | +0.432 |
| 19 | de licitação | +0.431 |
| 20 | naatz | +0.430 |
| 21 | ltda | +0.421 |
| 22 | empresas | +0.412 |
| 23 | licitações | +0.394 |
| 24 | investigação | +0.391 |
| 25 | prefeito | +0.388 |
| 26 | cge | +0.385 |
| 27 | esquema | +0.375 |
| 28 | nome da | +0.370 |
| 29 | dispensa de | +0.369 |
| 30 | dispensa de licitação | +0.366 |

---

## Próximos Modelos a Explorar (após fine-tuning do BERTimbau)

| # | Modelo | Arquitetura | Limite de Tokens | Origem / Referência | ID no Hugging Face |
|---|---|---|---|---|---|
| 0 | BERTimbau-base | BERT | 512 | PT-BR (Souza et al., 2020) | `neuralmind/bert-base-portuguese-cased` |
| 1 | NorBERTo | ModernBERT | 8.192 | PT-BR, 331B tokens (Silva et al., 2026) | `Itau-Unibanco/NorBERTo-base` |
| 2 | BERTugues | BERT | 512 | PT-BR (Mazza Zago & Pedotti, 2024) | (confirmar ID) |
| 3 | ModBERTBr | ModernBERT | 8.192 | PT-BR (Wu & Garcia, 2025) | `wallacelw/ModBERTBr` |
| 4 | multilingual-e5-large | XLM-RoBERTa (ajustado) | 512 | Multilíngue (intfloat) | `intfloat/multilingual-e5-large` |

### Motivação para modelos com limite > 512 tokens

A análise de truncamento no conjunto de treino revela que o teto de 512 tokens do BERT penaliza desproporcionalmente as notícias de fraude:

| Métrica | Notícias Normais (label=0) | Notícias de Fraude (label=1) | Interpretação |
|---|---|---|---|
| Amostras (n) | 36.088 | 562 | Severo desbalanceamento (~1,5% de fraudes) |
| Mediana | 382 tokens | 738 tokens | Metade das fraudes tem mais de 738 tokens — quase o dobro das normais |
| Q75 (75º percentil) | 605 tokens | 1.086 tokens | 25% das fraudes ultrapassam 1.000 tokens |
| % > 512 tokens | 33,2% | **70,6%** | Mais de 7 em cada 10 fraudes têm seu texto cortado |

> **Conclusão:** Modelos com janela de contexto de 8.192 tokens (NorBERTo, ModBERTBr) têm potencial de capturar evidências de fraude presentes na segunda metade dos textos, que o BERTimbau-base descarta completamente.
