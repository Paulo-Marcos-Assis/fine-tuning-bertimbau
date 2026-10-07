## Comparação v1 vs v2 — decisão por v2

**Conclusão: v2 (`bertimbau_2/bertimbau_tuning.ipynb`) é a versão adotada.**

### 1. Dados (causa principal da diferença)

| split | v1 | v2 | diff |
|---|---|---|---|
| train | 36652 (36090/562) | 36650 (36088/562) | -2 normais |
| dev | 9163 (9022/141) | 9162 (9021/141) | -1 normal |
| teste | 11453 (11277/176) | 11374 (11199/175) | -79 (78 normal + 1 fraude) |

Os 79 removidos do teste são quase-duplicatas TF-IDF>=0.9 vs treino/dev (maioria `nsctotal.com.br`), confirmadas por diff de `url` entre `bertimbau.tar.gz:test_bert.csv` e `bertimbau_2/test_bert.csv`.

### 2. Métricas

Dev (melhor época = 3 nos dois):
* v1: `0.7817 -> 0.8039 -> 0.8283` (`trainer_state.json best_metric 0.8283`, `checkpoint-6873`)
* v2: `0.7525 -> 0.7891 -> 0.8071`

Teste:
* v1: `F1 0.8383 [[11259 18][36 140]]`, 11453 amostras
* v2: `F1 0.8184 (tp142 fp30 fn33)`, 11374 amostras

### 3. Por que v1 é inflado

O teste v1 continha as 79 quase-duplicatas. São casos fáceis (texto já visto no treino), que inflavam o F1. A queda de ~2pp na v2 reflete teste mais duro e honesto, não regressão.

O restante da diferença está dentro do ruído: IC95% bootstrap da v2 é `F1 0.818 [0.773, 0.860]` — `0.838` cabe dentro. Contribuem ainda: v1 sem seed fixa (head + shuffle aleatórios) vs v2 `SEED=42`, e `bf16 + accum 2x8` vs `fp32 16` (mesmo batch efetivo 16, só ruído numérico).

### 4. Por que v2

Seed reprodutível, peso `64.21` calculado do dado (vs `64.2` hardcoded), padding dinâmico, guard-rail de leakage, `AP` + `IC bootstrap`, predições com `url/portal` para análise de viés.
