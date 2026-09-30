# Análise Exploratória — `finetuning/sets` (final padronizado + dedup)

**Data:** 2026-09-23  
**Base final:** `/home/paulo/CascadeProjects/Applied_ML/NEW_training/finetuning/sets/` — 3 arquivos BERT padronizados (`train_bert.csv`, `dev_bert.csv`, `test_bert.csv`) — origem `FOR_TRAINING/` + `FOR_TEST/Pre_processed_for_Embeddings/test_bert.csv` com dedup global.

## 1. Arquivos (após padronização e dedup)

| Arquivo | Origem | Linhas dados | wc -l (c/ header) | Colunas | Tamanho | md5 (pós-dedup) |
|---|---|---|---|---|---|---|
| `train_bert.csv` | `FOR_TRAINING/train.csv` | **36650** (-2) | 36651 | `url,processed_text,label,portal` | 82M | `f04fd2688ed590498b7289979910a618` |
| `dev_bert.csv` | `FOR_TRAINING/dev.csv` | **9162** (-1) | 9163 | `url,processed_text,label,portal` | 20M | `84a77c3491a79d65b3ec0ee4daf7196e` |
| `test_bert.csv` | `FOR_TEST/Pre_processed_for_Embeddings/test_bert.csv` | **11453** (-1) | 11454 | `url,processed_text,label,portal` | 26M | `72967052b3c2c2f7ccf05ede90f6f4bf` |

> `processed_text = title + " " + text` normalizado (`\n`, `\xa0`, `\s+` → ` `) com URLs (`https://`, `www.`, `bit.ly`, `t.co`) removidas. Validado: 98.3% idêntico após normalização, 1.7% restante por strip de URLs. `tuning.ipynb:141-146` renomeia `processed_text → text` e filtra `[[text,label]]`. **Dedup global por `md5(norm(processed_text))` removeu 4 linhas (0.007%) mantendo primeira ocorrência na ordem train→dev→test.**

**Antes do dedup:** 36652 + 9163 + 11454 = 57269. **Removidos 4** → **57265 final.**

## 2. Mapeamento validado

* `FOR_TRAINING/train.csv (6 cols: url,title,text,label,portal,date)` ↔ `sets/train_bert.csv (4 cols)` — **36650 linhas pós-dedup (original 36652, -2 intra), URLs 100% iguais e mesma ordem, label/portal 0 mismatches**
* `FOR_TRAINING/dev.csv` ↔ `sets/dev_bert.csv` — **9162 linhas pós-dedup (original 9163, -1 cross), 0 mismatches**

## 3. Volume combinado (final)

```
Total = 36650 (train) + 9162 (dev) + 11453 (test) = 57265 linhas (-4)
train 64.00% | dev 16.00% | test 20.00%
Supervisionado (train+dev) = 45812 (80.00%) → split 80.00/20.00 (train 80.00% / dev 20.00%)
```

## 4. Distribuição de classes (`label`: 0=NORMAL, 1=FRAUDE) — final

| Set | 0 | 1 | % fraude | Ratio |
|---|---|---|---|---|
| train | 36088 (-2) | 562 | 1.533% | 1:64.2 |
| dev | 9021 (-1) | 141 | 1.539% | 1:64.0 |
| test | 11277 (-1) | 176 | 1.537% | 1:64.1 |
| **train+dev** | **45109** (-3) | **703** | **1.535%** | **1:64.2** |
| **total** | **56386** (-4) | **879** | **1.535%** | **1:64.1** |

Distribuição idêntica (estratificação preservada). Removidos todos `label 0`, taxa inalterada. `tuning.ipynb:769` peso `64.2` permanece válido.

## 5. Qualidade / Integridade (final)

* **Duplicatas:** `url` duplicado = 0; linhas duplicadas = 0; `processed_text` duplicado = 0 (era 2+0+1)
* **Nulos:** `url,processed_text,label,portal` = 0 nos 3 BERT. No `FOR_TRAINING` cru: `title` 5+2 nulos, `date_publication` 99.1% NaN.
* **Overlap URLs:** `train∩dev = 0`, `train∩test = 0`, `dev∩test = 0` ✓
* **Overlap conteúdo `norm(processed_text)` após dedup:** `train∩dev = 0`, `train∩test = 0`, `dev∩test = 0` ✓ (antes: 1 cross train↔dev)
* **Ordem:** URLs mesma ordem entre `FOR` e `sets`

## 6. Portais (final)

* União total: **29** domínios
* `train 23 | dev 15 | test 19`
* Top 3: `ndmais.com.br ~79%` (28985/7351/9110), `www.nsctotal.com.br ~18%` (6877/1619/2108), `www.bbc.com` (263/68/64) — dev perdeu 1 nsctotal no dedup
* Lista completa: `agenciabrasil.ebc.com.br, agoralaguna.com.br, assinaturas.gazetadopovo.com.br, bsky.app, conta.gazetadopovo.com.br, especiais.gazetadopovo.com.br, f5.folha.uol.com.br, g1.globo.com, guia.folha.uol.com.br, iclnoticias.com.br, intranet.mpsc.mp.br, jornalconexao.com.br, lerunica.com.br, ndmais.com.br, olharsc.com.br, omunicipio.com.br, omunicipioblumenau.com.br, pc.sc.gov.br, portal.mpsc.mp.br, temas.folha.uol.com.br, www.assinecarta.com.br, www.bbc.com, www.cartacapital.com.br, www.cnnbrasil.com.br, www.gazetadopovo.com.br, www.mpsc.mp.br, www.nsctotal.com.br, www.tcesc.tc.br, www1.folha.uol.com.br`

## 7. Texto (`processed_text` chars) — final

| Set | min | q25 | mediana | média | q75 | max |
|---|---|---|---|---|---|---|
| train | 146 | 1030 | 1650 | 2127 | 2646 | 46194 |
| dev | 300 | 1026 | 1651 | 2094 | 2613 | 29305 |
| test | 71 | 1018 | 1649 | 2098 | 2610 | 40795 |

Inalterado (removidos 4 textos medianos). `max > 512 tokens` → truncamento `tuning.ipynb:283` (`max_length=512`).

## 8. Dedup aplicado (4 linhas)

* train: `https://www.gazetadopovo.com.br/vozes/polzonoff/michelle-bolsonaro-posse-tse` (sem `/`, duplicata de `.../tse/`, hash `bc20071b`)
* train: `https://www.nsctotal.com.br/noticias/fruta-jambo-tem-acao-preventiva...` (duplicata de `fruta-da-sua-infancia...`, hash `b14a4b47`)
* dev: `https://www.nsctotal.com.br/noticias/unimed-grande-florianopolis-nova-gestao-reposiciona-cooperativa-para-o-futuro-da-saude` (duplicata cross de train `...-saude-2`, hash `ec329224`)
* test: `https://www.assinecarta.com.br/?utm_medium=header` (duplicata de `?utm_medium=footer`, hash `1880933a`)

Critério: `hash = md5(re.sub(r'\s+',' ',processed_text).strip())`, manter primeira ocorrência na ordem train→dev→test.

## 9. Limites da auditoria exata e checagens complementares (`checagem_leakage.py`)

**O que foi garantido:** sem duplicatas **exatas** ou normalizadas por espaços/caixa (0 intra e 0 entre splits pós-dedup). Isso **não é** "zero vazamento".

* **URLs canônicas:** básica (barra/query/www) 0 cross; agressiva (+`-2`) **15 grupos entre splits** (8 test+train, 7 dev+train) — majoritariamente séries `parte-1/parte-2`, `ep-12/13`, `lotofacil 3555/3566`, `lua`, `cinema`. Indica molde repetido entre splits, revisar `saida_checagem/urls_canonicas_agressiva.csv`.
* **Quase-duplicatas TF-IDF (vizinho mais próximo):** dev→train ≥0.8: **173** (2 pos), ≥0.9: 63, ≥0.95: 18; test→train+dev ≥0.8: **240** (6 pos, 2 rótulos diferentes, 1 portal diferente). Top `1.0` são `parte-1/parte-2`; sim 0.95+ exige revisão manual — mediana pos 0.31/0.33, não-relacionados ~0.1-0.2, 5% palavras trocadas já derruba para ~0.80-0.86. Ver `quase_duplicatas_*.csv`.
* **Portal × fraude (atalho de fonte):** `ndmais 0.90%` (45446) e `nsctotal 0.44%` vs `iclnoticias 100%` (261), `g1 100%` (62). Fora dos 2 maiores: **2.1% das linhas concentram 48% dos positivos**. Baseline só-portal (logreg, limiar no dev): **test F1 0.586 AP 0.462** vs prevalência 0.015 — se BERT se aproximar disso, está usando fonte, não conteúdo.
* **Truncamento 512 tokens (`bert-base-portuguese-cased`):** mediana 381-385 tokens, q75 ~600-610. **Cortados: 33.6% geral, mas 70.6% train-pos / 73.0% dev-pos / 74.4% test-pos vs ~32-33% neg.** `tuning.ipynb:283` `max_length=512` esconde fim da matéria. Ver `saida_checagem/truncamento_por_classe.csv`.

Para texto da pesquisa: escrever "sem duplicatas exatas (exata/normalizada 0)" e reportar riscos acima como limitações.

## 10. Observações

* Split 64/16/20 preservado após dedup (80/20 train/dev).
* **Sem duplicatas exatas** após dedup; pronto para `DatasetDict(train=36650, validation=9162)` + holdout `test=11453`, mas avaliar riscos de quase-duplicatas, portal e truncamento.
* Imbalance ~1.535% exige `WeightedLoss (64.2)` e `F1` já configurados. Checagens completas em `saida_checagem/` e `checagem_leakage.py`.

## 11. Evolução dos dados por etapa (registro cumulativo)

As seções 1–10 descrevem a etapa 2. Cada etapa abaixo parte da anterior; nenhuma substitui o registro anterior.

| Etapa | Critério | train | dev | test | total |
|---|---|---|---|---|---|
| 1. Bruto | — | 36652 | 9163 | 11454 | 57269 |
| 2. Dedup exato/normalizado | `md5(norm(processed_text))`, train→dev→test (-4 linhas, todas `label 0`) | 36650 | 9162 | 11453 | 57265 |
| 3. Filtro de quase-duplicatas | sim. cosseno TF-IDF ≥ 0.9 com vizinho em train+dev; só no teste (-79) | 36650 | 9162 | **11374** | **57186** |

**Proporção entre splits (train / dev / test):**

| Etapa | % train | % dev | % test |
|---|---|---|---|
| 1. Bruto | 64.00 | 16.00 | 20.00 |
| 2. Dedup exato | 64.00 | 16.00 | 20.00 |
| 3. Filtro ≥ 0.9 | 64.09 | 16.02 | 19.89 |

**Proporção de fraude (label 1) por split:**

| Etapa | train | dev | test | total |
|---|---|---|---|---|
| 1. Bruto | 562 / 36652 = 1.533% | 141 / 9163 = 1.539% | 176 / 11454 = 1.537% | 879 / 57269 = 1.535% |
| 2. Dedup exato | 562 / 36650 = 1.533% | 141 / 9162 = 1.539% | 176 / 11453 = 1.537% | 879 / 57265 = 1.535% |
| 3. Filtro ≥ 0.9 | 562 / 36650 = 1.533% | 141 / 9162 = 1.539% | 175 / 11374 = 1.539% | 878 / 57186 = 1.535% |

Na etapa 3 saíram 79 linhas do teste (78 normais, 1 fraude): a estratificação foi preservada e o peso da classe (64.2, calculado do treino) não muda. md5 do teste na etapa 3: `86029ea0...` (detalhes em `AUDITORIA_DATALEAKAGE.md`, seção 7).

**Teste na etapa 3 (chars):** | Set | min | q25 | mediana | média | q75 | max |
| test | 71 | 1017 | 1650 | 2100 | 2616 | 40795 |

Truncamento >512 tokens no teste: normal 33.0% (mediana 380 tokens), fraude 74.3% (mediana 712).
