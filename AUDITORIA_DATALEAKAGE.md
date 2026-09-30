# Auditoria Anti-DataLeakage — `finetuning/sets` (padronizado + dedup final)

**Data:** 2026-09-23  
**Base final:** `/home/paulo/CascadeProjects/Applied_ML/NEW_training/finetuning/sets/` — `train_bert.csv` (36650), `dev_bert.csv` (9162), `test_bert.csv` (11453) — total **57265** linhas  
**Padronização:** header `url,processed_text,label,portal` utf-8 `label int64` em todos. `test_bert.csv` copiado e reescrito para mesma ordem/colunas. `md5` final: `train f04fd268...`, `dev 84a77c34...`, `test 72967052...`  
**Dedup global:** `md5(norm(processed_text))` onde `norm = re.sub(r'\s+',' ',s).strip()`, ordem de prioridade train→dev→test, manter primeira ocorrência.

## Método

1. **URL exato** — `set(url)` interseção
2. **Conteúdo exato** — `md5(processed_text)`
3. **Conteúdo normalizado** — `md5(norm(text))`
4. **Conteúdo lower** — `md5(norm(text).lower())`
5. **Intra-set** — `duplicated(subset=['processed_text'])`

## 1. Estado ANTES do dedup (para registro)

| Par | URL | Texto exato | Texto norm | Texto lower |
|---|---|---|---|---|
| train∩dev | 0 ✓ | **1** ✗ | **1** ✗ | **1** ✗ |
| train∩test | 0 ✓ | 0 ✓ | 0 ✓ | 0 ✓ |
| dev∩test | 0 ✓ | 0 ✓ | 0 ✓ | 0 ✓ |
| **Intra** | — | train 2 pares (4 linhas) | — | — |
| | — | test 1 par (2 linhas) | — | — |
| | — | dev 0 | — | — |
| **Total únicos** | 57269/57269 URLs | 57265/57269 textos (4 duplicatas) |

Detalhe das 4 duplicatas removidas (8 linhas):

* **Cross train↔dev hash `ec329224`** `len 7909`: train `.../unimed-...-saude-2` vs dev `.../unimed-...-saude` — `Unimed Grande Florianópolis: nova gestão reposiciona...`
* **Intra train `b14a4b47`** `len 3183`: `fruta-da-sua-infancia...` vs `fruta-jambo...` — `Fruta jambo tem ação preventiva...`
* **Intra train `bc20071b`** `len 3293`: `.../michelle-bolsonaro-posse-tse/` vs `.../tse` — `Você cumprimentaria (com beijinhos) o carrasco...`
* **Intra test `1880933a`** `len 413`: `assinecarta ...?utm_medium=footer` vs `?utm_medium=header` — `Junte-se a nós! Respeitamos...`

## 2. Ação de dedup (executada)

```python
seen = {}
for set in [train, dev, test]:
  for row in set:
    h = md5(norm(row.processed_text))
    if h in seen: remover(atual)
    else: seen[h]=row.url
```

Removidos 4 linhas (0.007%):

* train: `.../michelle-bolsonaro-posse-tse` (sem `/`) — duplicata de `.../tse/` (linha 23472)
* train: `.../fruta-jambo...` — duplicata de `.../fruta-da-sua-infancia...` (linha 32766)
* dev: `.../unimed-...-saude` — duplicata cross de train `...-saude-2` (dev linha 523) — manter train
* test: `...assinecarta...header` — duplicata de `...footer` (test linha 9304)

## 3. Estado FINAL pós-dedup (atual em `sets/`)

| Par | URL | Texto exato | Texto norm | Texto lower | Status |
|---|---|---|---|---|---|
| train∩dev | 0 | 0 | 0 | 0 | ✓ |
| train∩test | 0 | 0 | 0 | 0 | ✓ |
| dev∩test | 0 | 0 | 0 | 0 | ✓ |
| train intra | 0 dup_url, 0 dup_text | 0 | 0 | 0 | ✓ |
| dev intra | 0 | 0 | 0 | 0 | ✓ |
| test intra | 0 | 0 | 0 | 0 | ✓ |
| **Total** | **57265/57265 URLs** | **57265/57265 textos** | **57265** | **57265** | **✓ 100% único** |

```
train 36650 (36088×0, 562×1) | dev 9162 (9021×0, 141×1) | test 11453 (11277×0, 176×1)
Total 57265 = 36650+9162+11453
```

## 4. Conclusão — alcance correto

* **URLs exatas:** ✓ 0 entre splits antes e depois.
* **Conteúdo exato/normalizado (md5 `norm` e `lower`):** ✗ antes com 1 cross + 3 intra; **✓ após dedup: 0** — **"sem duplicatas exatas"**, não "zero vazamento".
* **O que ainda pode vazar (checagem_leakage.py, 57265 linhas):**
  * **Canônico agressivo** (sem `www`, barra, query, `-2`): 29 grupos, **15 entre splits** (8 test+train, 7 dev+train) — séries `parte-N/ep-N`, `lotofacil`, `lua`, `cinema`. Ver `urls_canonicas_agressiva.csv`.
  * **TF-IDF cosseno (vizinho mais próximo):** dev→train ≥0.8: 173 (2 pos), test→train+dev ≥0.8: 240 (6 pos, 2 rótulos diferentes, 1 portal diferente), ≥0.90: 63/79. Mediana pos 0.31-0.33 vs não-relacionados 0.1-0.2. Top são templates `parte-1/2`. Quase-duplicatas de mesmo fato em portais diferentes apareceriam com `portais_diferentes` alto — aqui 0-1, mas exige leitura manual de `quase_duplicatas_*.csv`.
  * **Portal (atalho):** 2 maiores portais têm 0.44-0.90% fraude, mas `iclnoticias/g1 100%`; 2.1% das linhas fora top2 concentram 48% dos positivos. Baseline só-portal: **F1 0.586 AP 0.462** (vs aleatório 0.015). Se BERT ≈ isso, usa fonte.
  * **Truncamento 512:** 33.6% cortados geral, **~73% dos positivos vs ~33% dos negativos** (mediana pos ~730 tokens vs neg ~381). Medida real com `bert-base-portuguese-cased` em `truncamento_por_classe.csv`. Se sinal de fraude está no fim, modelo não vê.

## 5. Recomendação pós-dedup

* Usar splits finais `36650/9162/11453` com rótulo **"sem duplicatas exatas"** no texto da pesquisa.
* Não afirmar "métricas não infladas" — reportar como limitação os 15 grupos canônicos, 173/240 pares ≥0.8 e baseline de portal.
* Manter pipeline de dedup global por `norm_hash` + rodar `checagem_leakage.py` a cada novo split; para mitigar portal, avaliar mascarar portal ou estratificar por portal; para truncamento, considerar `longformer`/`hierarchical` ou `max_length>512`.

## 6. Padronização confirmada (final)

```
cols: ['url','processed_text','label','portal'] em todos
dtypes: label int64, resto StringDtype
nulos: 0 | dup_url: 0 | dup_rows: 0 | dup_text_norm: 0
```

## 7. Filtro de quase-duplicatas no teste (aplicado)

* Critério: similaridade de cosseno TF-IDF (`checagem_leakage.py`) ≥ **0,9** entre a linha do teste e seu vizinho mais próximo em train+dev.
* Removidas **79** linhas de `test_bert.csv` (0,7%), das quais 1 positiva: **11453 → 11374** (11277→11199 negativos, 176→175 positivos). Train e dev não foram alterados. Novo md5 do teste: `86029ea0...`.
* Script: `bertimbau_2/filtra_quase_duplicatas.py` (lê `saida_checagem/quase_duplicatas_test.csv`, gerado antes do filtro). Os números das seções 1–6 referem-se ao teste original.
* Limitação: pares com similaridade entre 0,8 e 0,9 (161 no teste) permanecem; o corte de 0,9 é arbitrário.
