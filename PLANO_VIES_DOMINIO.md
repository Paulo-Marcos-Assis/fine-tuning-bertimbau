# Plano — Teste de viés de domínio (portal) no fine-tuning do BERTimbau

**Data:** 2026-09-30 · **Status:** PLANEJAMENTO — nada deste plano foi implementado ainda.
**Público-alvo:** agentes/colaboradores que vão implementar as tasks abaixo. Leia as seções 1–3 antes de qualquer task.

## 0. Regras de trabalho (valem para qualquer agente)

1. **Não fazer commit nem push sem autorização explícita do usuário.** O fluxo é: editar no clone local → o usuário faz push → `git pull` no Jupyter (https://jupyter.ceos.ufsc.br, GPUs RTX A6000) só nos momentos de execução pesada.
2. **O conjunto de teste (`bertimbau_2/test_bert.csv`, 11.374 linhas) não pode influenciar nenhuma decisão** (hiperparâmetro, limiar, checkpoint, escolha de modelo). Limiares e seleção saem só do dev.
3. Artefatos novos vão em pastas novas (sugestão: `bertimbau_2/vies_dominio/`). **Exceção decidida em 2026-09-30:** `train/dev/test_bert.csv` foram sobrescritos pela limpeza do rodapé ICL (seção 1.4); as versões anteriores estão no histórico do git (`git show e28a331:bertimbau_2/train_bert.csv`, idem `dev`; teste: `git show 26129f6:bertimbau_2/test_bert.csv`).
4. Antes de treinar na GPU, conferir `nvidia-smi` e ajustar `CUDA_VISIBLE_DEVICES` (em 2026-09-30 as GPUs 0 e 1 estavam livres; a 2 tinha 44 GB ocupados por outro usuário).
5. Documentar cada etapa como **evolução** (não substituir o registro anterior) em `AUDITORIA_DATALEAKAGE.md` / `ANALISE_EXPLORATORIA.md`. Texto objetivo — vai servir de base para o artigo.
6. Idioma: português.
7. **Pontos com confirmação pendente na implementação:** peso de classe por configuração e critério de escolha da razão vencedora (seção E3). Pedir confirmação ao usuário ao chegar neles.

## 1. Contexto

### 1.1 O repositório atual
- Fine-tuning de `neuralmind/bert-base-portuguese-cased` para detectar notícias de fraude/irregularidade (label 1) vs. normais (label 0). Notebook: `bertimbau_2/bertimbau_tuning.ipynb` (loss ponderada, seed 42, bf16, 3 épocas, seleção do checkpoint por F1 no dev, avaliação final única na PARTE 9).
- Dados (`bertimbau_2/*.csv`, colunas `url, processed_text, label, portal`), estado atual após as etapas de `ANALISE_EXPLORATORIA.md` seções 11–12:

| Split | Linhas | Positivos | % fraude |
|---|---|---|---|
| train | 36.650 | 562 | 1,533% |
| dev | 9.162 | 141 | 1,539% |
| test | 11.374 | 175 | 1,539% |

- Origem dos dados: pipeline em `/home/paulo/CascadeProjects/Applied_ML/NEW_training/` (mesmos splits; lá o teste tinha 11.454 linhas antes da deduplicação e do filtro de quase-duplicatas).

### 1.2 O problema: o portal está confundido com a classe
Contagens recalculadas nos CSVs atuais (`bertimbau_2/*.csv`, teste já filtrado):

| Grupo | Linhas | Positivos | % fraude |
|---|---|---|---|
| `ndmais.com.br` | 45.427 | 410 | 0,9% |
| `nsctotal.com.br` | 10.546 | 47 | 0,45% |
| Todos os outros (28 domínios, inclui `nsctotal`) | 11.759 | 468 | 4,0% |

- Fora dos 2 maiores portais: 2,1% das linhas concentram **48% dos positivos**.
- `iclnoticias.com.br` (261 linhas, 100% fraude), `g1.globo.com` (62, 100%) e `jornalconexao.com.br` (76, 53%) somam 363 positivos (**41% de todos os positivos**). Portais institucionais (`mpsc`, `pc.sc.gov.br`, `tcesc`) são 100% fraude mas têm ~10 exemplos.
- Assinatura da fonte no texto: o termo "ICL Notícias" aparece em 32 textos, **100% positivos** (depois da limpeza do rodapé); "jornal conexão" 4/5 positivos; "g1" 12,4% positivos. Vale testar mascaramento (T4).

### 1.3 Outros riscos já documentados (não são o foco deste plano, mas interagem)
- Truncamento em 512 tokens atinge ~74% dos positivos vs ~33% dos negativos (`saida_checagem/truncamento_por_classe.csv`).
- Quase-duplicatas (séries `parte-1/2`, agenda de candidatos, cinema, lua): removidas do teste só acima de 0,9 de cosseno TF-IDF; restam 161 pares entre 0,8 e 0,9.

### 1.4 Limpeza do rodapé do ICL Notícias (feita)
- Padrão: `Redação ICL Economia…Com informações da/do …` colado no fim do texto (ex.: `…Redação ICL EconomiaCom informações da Folha de S.Paulo`). Regex em `bertimbau_2/limpa_assinatura_icl.py`, aplicado a **todos os portais** (só o `iclnoticias` tem esse padrão): 38 textos alterados (26 treino, 5 dev, 7 teste); a seção "Relacionados" posterior, quando existe, foi mantida.
- Escolha do usuário: limpar **apenas** esse rodapé. **Não** foram tratados: menções a ICL no corpo (ex.: "ICL Notícias teve acesso…", 32 textos), byline "Por <Nome> —" no início (45 textos, só 1 fora do ICL) e créditos de sindicação. ~70% dos textos do ICL não têm nenhuma assinatura explícita, então o estilo/assunto ainda pode identificar a fonte.
- Não altera contagens de linhas nem rótulos; md5 novos em `ANALISE_EXPLORATORIA.md` seção 12.

### 1.5 `iclnoticias` não é portal de treino do E3, mas entra na exposição com quantidade limitada (decisão do usuário, 2026-09-30)
- O portal tem 261 positivos e **nenhum negativo**. Como portal de treino, "estilo ICL" viraria um atalho perfeito para "fraude" (treino: 160 de 160) e os negativos do treino viriam só do `ndmais`. Por isso **não entra no treino nem no dev do E3**.
- No conjunto de exposição (portais que o modelo nunca viu) ele **entra**, mas com teto por portal (~11 positivos, não os 261), para não dominar os positivos do conjunto. Como só tem positivos, ele só contribui para o recall agregado.
- Continua no dataset e no treino do notebook principal.


## 2. O experimento de referência (outro classificador): NDMAIS_BIAS

Local: `/home/paulo/CascadeProjects/Applied_ML/to_zip/NDMAIS_BIAS/` (302 MB; caminhos absolutos dentro dos scripts apontam para `Applied_ML/NDMAIS_BIAS`, que não existe mais — ajustar ao reutilizar).

**Ideia:** treinar só com `ndmais.com.br` (positivos **e** negativos do mesmo portal → o portal deixa de ser confundidor), depois medir em outros portais e inspecionar os pesos do modelo.

| Etapa | Onde | O que faz |
|---|---|---|
| Dataset ndmais | `ndmais_dataset/` | 10.693 linhas (729 pos / 9.964 neg, ~1:10) |
| Split/dedup/preproc | `scripts/task1*_*.py`, `task2_split.py`, `task3_preprocess.py` | train 5.173 / dev 1.292 / test 1.616 |
| 12 classificadores | `scripts/task4_vectorize.py`, `task5_train.py`, `training/results/` | 3 vetorizações (TF-IDF, BERTimbau base/large embeddings) × 4 (LinearSVC, LR, RF, XGBoost); `ALL_dev_results.csv` |
| Teste in-domain | `scripts/task7_final_test.py` | F1 = 1,0 (SVC/LR), 0,993 (RF) — **suspeito de tão perfeito** |
| Viés de domínio (Task 8) | `scripts/task8_domain_bias.py`, `ndmais_doc/DOMAIN_BIAS_REPORT.md` | top-20 coeficientes TF-IDF (SVM/LR), classificados como "semântico" vs "estilístico do portal" por lista de palavras; concluiu "viés BAIXO" (95% semântico) |
| Cross-portal | `CROSS_PORTAL_TEST/` (`PLANO.md`, `build_subsets.py`, `run_inference_all12.py`, `results_all12/REPORT_ALL12.md`) | 5 subsets disjuntos de outros portais (39 pos + 399 neg cada, razão do teste ndmais), só inferência, sem refit |

### 2.1 Lições do experimento (aplicar aqui)
1. **Task 8 sozinha não prova ausência de viés.** Com dataset de um único portal, "cross-domain N/A"; a classificação por lista de palavras é heurística. O teste que de fato revelou algo foi o cross-portal.
2. **Resultado cross-portal (2.190 linhas, 195 pos):** TF-IDF + LinearSVC manteve F1 0,965 (Δ +0,004 vs dev); já BERTimbau (embeddings + cabeça linear) caiu para **F1 0,36–0,40** apesar de recall 0,99 e PR-AUC ~0,91–0,93. Ou seja: **o ranking transferiu, o limiar não** (precisão ~0,22, superprevisão de positivos). → Usar métricas independentes de limiar (AP/PR-AUC) como principais e escolher o limiar no dev; nunca reportar só F1 a 0,5.
3. **F1 por portal é enganoso:** portais 100% positivos dão F1 = 1 trivial; portais sem positivos dão F1 = 0. → No E3 isso não se aplica (decisão do usuário: só métrica **agregada**, como no experimento de referência); a lição vale para o E1, onde, se houver quebra por portal, usar recall/FPR e nunca F1.
4. O pool cross-portal de lá era pequeno (199 pos, 59 portais; 195 usados em 5 subsets com 8,9% de fraude). Aqui o pool de origem (todo portal que não é o `ndmais`) tem **468 positivos em 21 portais**, mas 78% vêm de 3 fontes (`iclnoticias` 261, `g1` 62, `jornalconexao` 40; só o `iclnoticias` tem 56%). Por isso os positivos do conjunto de exposição são **estratificados por portal** (E3), para nenhuma fonte dominar.

## 3. Baseline de comparação (confirmação do F1 0,7420)

**Confirmado**, com ressalvas. Em `/home/paulo/CascadeProjects/Applied_ML/NEW_training/`:
- **Experimento B1 = TF-IDF + LinearSVC + random oversampling 1:10 (dentro de cada fold), sem `class_weight`**, TF-IDF n-grama (1,3), 20k features, `C=0.1`, CV 5-fold estratificado. Resultados em `training/results/tfidf/svm/experiment_b1_oversampling/` (`COMPARISON.md`, `experiment_b1_results.json`, `model.pkl`, script `experiment_b1_oversampling.py`; vetorizador em `experiment_a/vectorization/vectorizer.pkl`). Narrativa em `new_chronology/PROGRESS.md` (Task 6).
- Métricas **no dev**: F1 **0,7420**, precision 0,7394, recall 0,7447, PR-AUC 0,7882, ROC-AUC 0,9926. Supera o baseline da Task 5 (F1 0,7075, que é o "melhor" citado em `training/results/CONSOLIDACAO_FINAL.md`) e o undersampling B2 (0,7010).
- **Ressalvas:**
  - É F1 no **dev** (9.163 linhas, anterior à dedup de 1 linha), e a configuração foi escolhida comparando no dev. **Não há avaliação do B1 no teste** (a Task 7 do NEW_training não foi executada para ele).
  - O TF-IDF usa o texto do **pré-processamento pesado** (sem acento, minúsculas, sem stopwords; `NEW_training/FOR_TEST/Pre_processed_for_Sparse`, `vectorization/test_preprocessed.csv`), não o `processed_text` do BERT. A ligação entre os dois é a `url`.
  - Treinado no treino pré-dedup (36.652 linhas) e o teste de lá tem 11.454 linhas.

## 4. Decisões recomendadas (responde às dúvidas do usuário)

**4.1 Comparar com 1 baseline em vez de 12 modelos: sim.** O objetivo aqui é medir o fine-tuning, não repetir a grade de classificadores. Comparar (a) o BERTimbau fine-tuned e (b) **um** baseline de conteúdo = TF-IDF + LinearSVC (config B1). Os 12 modelos do NDMAIS_BIAS não entram (o experimento de lá usa embeddings congelados + cabeça linear, que é outra família e colapsou no cross-portal).

**4.2 Quando fazer — em três níveis, por custo:**

| Nível | O que é | Quando | Custo |
|---|---|---|---|
| **E1** | Quebrar as métricas do modelo **já treinado** por portal (ndmais vs. fora do ndmais; recall/FPR por portal) | **Depois** da execução principal, sem retreinar | ~zero, mas exige salvar predições (ver abaixo) |
| **E2** | Reimplementar o baseline TF-IDF+SVM (B1) neste repo, com o mesmo teste | **Em paralelo agora** (CPU, local, minutos) | baixo |
| **E3** | Treinar só com o texto (título + corpo) do `ndmais` em 3 razões de classe (natural 1:106, 1:64 e 1:10, por subamostragem de negativos), escolher a melhor no dev e expor esse modelo a outros portais — reprodução do NDMAIS_BIAS — para o BERT e para o baseline | **Depois** da execução principal, como 2ª rodada na GPU | 3 fine-tunings extras (o de 1:10 é pequeno: ~3 mil linhas) |
| E4 (opcional) | Mascarar assinaturas de portal no texto (ex.: "ICL Notícias") e reavaliar | Só se E1/E3 indicarem atalho | baixo |

**Única coisa a fazer antes de rodar o notebook principal:** acrescentar uma célula que **salva as predições** (probabilidade da classe fraude + `url` + `label`) de dev e teste em CSV. Sem isso, E1 exigiria retreinar. A execução principal não muda em mais nada e não depende do resultado de E1–E4.

## 5. Desenho dos experimentos

### E1 — Métricas por portal (sem retreino)
- Entrada: predições salvas (`url, portal, label, p_fraude`) do BERT e do baseline.
- Cortes: `ndmais` × `iclnoticias` × `nsctotal` × "outros"; tabela por portal com n, positivos, **recall, FPR**, AP do grupo. (No E1 o modelo principal foi treinado em todos os portais, então `iclnoticias` entra na tabela; só não entra no E3.)
- Limiar: escolhido no **dev** (F1 máx.) e aplicado ao teste; reportar também AP (sem limiar).
- IC por bootstrap (2.000 reamostras) — positivos são poucos (teste: 175; fora do ndmais no teste: 94).

### E2 — Baseline TF-IDF + LinearSVC (config B1)
- Reimplementar o script B1 lendo os CSVs atuais (ou reaproveitar `vectorizer.pkl`/`model.pkl` e prever no teste filtrado por `url`). Hiperparâmetros fixos da config B1; não re-tunar no teste.
- Saída no **mesmo formato** de predições do BERT, para o E1 tratar os dois igual.

### E3 — Treino só no ndmais (3 razões) → exposição a outros portais
Reprodução do experimento de referência, com as decisões do usuário (2026-09-30): treino **só com o `ndmais`** (positivos e negativos do mesmo portal, então o portal deixa de ser confundidor); **três razões de classe no treino**; **um único conjunto de exposição** (em vez de 5 subsets); **métricas só agregadas**; e a razão da exposição **segue a do treino vencedor**. O nome do portal **não** é entrada do modelo.

**Treino: 3 configurações** (negativos subamostrados ao acaso, seed 42; os 270 positivos ficam todos):

| Config | Razão (neg:pos) | Positivos | Negativos | Linhas | Como chega lá |
|---|---|---|---|---|---|
| Natural (ponto de partida) | **1:106** (valor atual; treino 28.715/270) | 270 | 28.715 | 28.985 | sem subamostragem |
| 1:64 | 1:64 (razão do melhor baseline do NEW_training) | 270 | 17.280 | 17.550 | subamostra 11.435 negativos |
| 1:10 | 1:10 (razão do NDMAIS_BIAS e do B1) | 270 | 2.700 | 2.970 | subamostra 26.015 negativos |

- Dev (`ndmais`: 7.351 linhas, 59 pos, razão 1:124) e teste in-domain (9.091 linhas, 81 pos, 1:111) ficam **sempre na proporção natural**, para todas as configurações.
- Rodar as 3 configurações para o **BERT** e para o **baseline TF-IDF + SVM** (config B1, sem o oversampling, já que a razão vem da subamostragem).
- **Peso de classe (confirmado pelo usuário em 2026-09-30):** 1:64 e 1:10 sem peso na loss (a subamostragem é o ajuste, como no B2); natural com o peso do notebook principal (neg/pos ≈ 106).
- **Escolha da melhor configuração (confirmado em 2026-09-30):** no dev do `ndmais`, por AP; F1 com o limiar do dev como critério secundário. Escolhe-se por modelo (BERT e baseline), sem olhar a exposição.
- **Ressalva:** o usuário vai estudar depois a justificativa técnica/teórica desses dois pontos. **Ao chegar nas tasks T6, T7 e T7b, o agente deve pedir confirmação ao usuário antes de implementá-los.**

**Pool de origem da exposição:** todo portal que não é o `ndmais` (inclui `iclnoticias` e `nsctotal`): 11.759 linhas, 21 portais com positivos, 468 pos / 11.291 neg. Nunca visto no treino/dev. Conferido: a similaridade TF-IDF máxima de qualquer linha do pool com o treino do `ndmais` é 0,65, então o filtro ≥ 0,9 não remove nada.

**Conjunto de exposição (amostrado com seed 42; razão = a do treino vencedor):**
1. Filtro de similaridade ≥ 0,9 contra o treino `ndmais` (hoje remove 0 linhas) e checagem de `url` em comum (deve ser 0).
2. **Positivos estratificados por portal** com teto por portal, para nenhuma fonte dominar (o `iclnoticias` não usa os 261). Portais com poucos positivos entram com todos.
3. **Negativos sorteados** do pool de negativos até a razão do treino vencedor.
4. Como os negativos limitam (11.291), o teto é o maior que cabe na razão. Os conjuntos são aninhados (mesma ordem sorteada), para as exposições serem comparáveis:

| Razão da exposição | Teto | Positivos | Negativos | Total | `iclnoticias` nos positivos |
|---|---|---|---|---|---|
| 1:106 (natural) | 12 | 106 | 11.273 | 11.379 | 11% |
| 1:64 | 29 | 174 | 11.136 | 11.310 | 17% |
| 1:10 | 29 (mesmos positivos) | 174 | 1.740 | 1.914 | 17% |

- Distribuição dos positivos no teto 29: `iclnoticias`, `g1`, `nsctotal` e `jornalconexao` com 29 cada; `bbc` 12; `cartacapital` 10; `gazetadopovo` 9; `agoralaguna` 7; `mpsc` 5; `folha` 3; `pc.sc.gov.br` 2; e 10 portais com 1. No teto 12 os quatro primeiros e o `bbc` ficam com 12.
- Negativos do pool: 93% `nsctotal` (10.499), 3% `bbc` (382), o resto com poucos casos; o sorteio mantém essa composição.

**Métricas (agregadas, no conjunto único):** AP/PR-AUC, precisão, recall, F1 com limiar escolhido no **dev `ndmais`**, ROC-AUC, matriz de confusão, taxa de positivos previstos vs. real, variação do F1 em relação ao teste in-domain do `ndmais`, e IC por bootstrap. Sem métrica por portal.
- **Limites a registrar:** poucos positivos (106–174, IC largo); ~93% dos negativos vêm do `nsctotal`; o `iclnoticias` contribui com positivos e nenhum negativo; só 270 positivos distintos no treino em qualquer configuração.
- Opcional (E3b, leave-one-portal-out): treinar o modelo completo sem `g1` (ou sem `jornalconexao`) e testar nele.
- Interpretação: se o modelo vencedor mantém AP e recall razoáveis na exposição, o conteúdo generaliza.

### E4 — Mascaramento de assinatura (opcional)
- Remover do texto nomes/bordões de portal (ex.: "ICL Notícias", "Jornal Conexão", "Leia também", sufixos de portal) e reavaliar só o baseline (barato); depois o BERT, se relevante.
- Para o BERT não há atribuição por coeficiente; usar a variação entre com/sem máscara. Para o baseline TF-IDF, listar os top-20 coeficientes e marcar termos de portal (análogo à Task 8, mas com lista de termos de portal derivada dos dados, não só palavras-chave de fraude).

### Métricas (todas as tasks)
- Primária: **AP/PR-AUC**. Secundárias: precision/recall/F1 **em limiar escolhido no dev**; matriz de confusão; IC bootstrap.
- E3: só métrica agregada (sem quebra por portal). E1: se houver quebra por portal, recall (positivos) e FPR (negativos), nunca F1 (ver 2.1.3).
- Comparar sempre o BERT com o baseline de conteúdo (TF-IDF + SVM) no mesmo teste.
- Se possível, 3–5 seeds no fine-tuning (o notebook fixa só a seed 42).

## 6. Tasks

Cada task termina com critérios de aceite. Marcar como feito neste arquivo (seção 8).

| # | Task | Depende de | Entregável | Aceite |
|---|---|---|---|---|
| T1 | **Salvar predições no notebook:** após `trainer.predict`, gravar CSV com `url, portal, label, p_fraude` para dev e teste (`bertimbau_2/vies_dominio/pred_bert_{dev,test}.csv`). Incluir também o dev (para o limiar). | — | célula nova no notebook | Linhas = 9.162 e 11.374; `url` alinhado aos CSVs; nenhum uso do teste antes da PARTE 9 |
| T2 | **Baseline TF-IDF+SVM (B1)** reproduzido no repo atual, com predições no mesmo formato (`pred_tfidf_{dev,test}.csv`). Decidir se reaproveita `model.pkl` do NEW_training ou retreina (recomendado: retreinar com a config B1 nos CSVs atuais, script versionado). | — | `bertimbau_2/vies_dominio/baseline_tfidf_svm.py` | F1 dev próximo de 0,74 (registrar a diferença vs. 0,7420 e o motivo); teste usado uma única vez |
| T3 | **Script de análise por portal (E1):** lê as predições, calcula métricas agregadas e por portal (recall/FPR), limiar do dev, bootstrap, tabelas `.csv` + resumo `.md`. | T1, T2 | `bertimbau_2/vies_dominio/analise_portal.py` | Gera `metricas_por_portal.csv` e `resumo.md`; nenhuma métrica por portal em F1 |
| T4 | **Mascaramento (E4) no baseline TF-IDF:** top-20 coeficientes com marcação de termos de portal + reavaliação com texto mascarado. Lista de termos de portal derivada dos dados (termos com alta concentração em um portal). | T2 | `mascara_portal.py`, seção no resumo | Tabela com/sem máscara (AP, F1 no limiar do dev) |
| T5 | **Preparar dados E3:** script que (a) separa `train/dev/test` só do `ndmais` e gera os 3 treinos (natural, 1:64, 1:10) por subamostragem de negativos (seed 42, positivos todos mantidos); (b) monta os conjuntos de exposição aninhados (1:106, 1:64, 1:10) com os passos da seção E3 (positivos por portal com teto, negativos sorteados, checagem de similaridade e de `url`); (c) reporta contagens por portal e por classe. `build_subsets.py` do NDMAIS_BIAS serve só de referência. | — | `bertimbau_2/vies_dominio/ndmais_only/` (`train_natural/1a64/1a10.csv`, `dev.csv`, `test.csv`, `exposicao_*.csv`) + relatório de contagens | Razões batem com a tabela (1:106,4 / 1:64 / 1:10); exposição sem `ndmais`, sem `url` em comum com o treino; nenhum portal acima do teto; nome do portal não entra como feature |
| T6 | **Notebook E3 (BERT):** cópia parametrizada do notebook principal, treinando só no `ndmais` nas 3 configurações (mesma config, seed 42; peso de classe conforme a seção E3), com predição no dev, no teste in-domain e na exposição, no formato de T1. Rodar no Jupyter (conferir GPU livre). | T1, T5 | `bertimbau_ndmais_only.ipynb` | 3 treinos concluídos; predições salvas; a exposição não usada para escolher checkpoint/limiar/configuração |
| T7 | **Baseline E3:** TF-IDF + SVM (config B1, sem oversampling) treinado só no `ndmais` nas 3 configurações, mesmas saídas. | T2, T5 | predições | idem |
| T7b | **Escolher a razão vencedora por modelo** no dev `ndmais` (critério da seção E3) e **avaliar só esse modelo** no conjunto de exposição da mesma razão; métricas agregadas com IC. | T6, T7 | tabela de dev (3 configs × 2 modelos) + resultado na exposição | A exposição só é lida depois da escolha; critério e resultado registrados |
| T8 | **(Opcional) E3b leave-one-portal-out** para `g1` e `jornalconexao`. | T6/T7 | tabela recall por portal retido | — |
| T9 | **Consolidar e documentar:** tabela final (modelo × {ndmais in-domain, exposição a outros portais}, AP, F1@dev-thr, recall, IC; só agregado), conclusões e limitações; atualizar `AUDITORIA_DATALEAKAGE.md`/`ANALISE_EXPLORATORIA.md` como nova etapa. Texto curto (vai para o artigo). | T3 + (T4,T6–T7b,T8) | seção nova nos `.md` | Inclui as limitações: 106–174 positivos na exposição, 93% dos negativos do `nsctotal`, e o `iclnoticias` só como exposição (11 positivos, sem negativos) e fora do treino |

**Ordem sugerida:** T1 (antes de rodar o notebook) → T2 em paralelo, localmente → rodar o notebook principal no Jupyter → T3 → decidir se o E3 vale o custo olhando o resultado de T3 → T5–T7 → T9. T4 e T8 só se necessário.

## 7. Riscos e pontos de atenção
- **Poucos positivos** → métricas instáveis; sempre IC e contagens absolutas. Diferenças de 2–3 pp de F1 são ruído.
- **Reuso de caminhos absolutos** dos scripts do NDMAIS_BIAS (`/home/paulo/CascadeProjects/Applied_ML/NDMAIS_BIAS`) — a pasta foi movida para `to_zip/`. Não copiar cegamente.
- **Texto do baseline ≠ texto do BERT** (pré-processamento pesado vs. leve): alinhar sempre pela `url`; conferir contagem de linhas após o join.
- **Escolha de limiar/configuração no teste** é o erro mais fácil de cometer aqui; o teste só para o resultado final de cada modelo.
- **Filtro de quase-duplicatas** foi aplicado só ao teste (≥ 0,9 vs. train+dev); no E3 o pool cross-portal precisa de filtro próprio contra o treino `ndmais` do E3.
- O F1 perfeito do NDMAIS_BIAS in-domain (1,0) e "viés BAIXO" da Task 8 **não devem ser citados como evidência** de ausência de viés; o que vale dali é o cross-portal e a lição do limiar.

## 8. Checklist de progresso
- [x] T1 (células adicionadas ao notebook em 2026-09-30; falta executar no Jupyter) · [ ] T2 · [ ] T3 · [ ] T4 · [x] T5 (concluída em 2026-10-01) · [ ] T6 · [x] T7/T7b (concluída em 2026-10-01: Baseline TF-IDF+SVM treinado nas 3 configs, vencedor selecionado no dev e avaliado na exposição; relatório e predições gerados) · [ ] T8 (opcional) · [ ] T9

## 8b. Tarefa futura: comparar outros modelos no fine-tuning
NorBERTo, ModBERTBr, BERTugues e multilingual-e5-large contra o BERTimbau: ver `PLANO_COMPARACAO_MODELOS.md` (não iniciada).

## 9. Mapa de referências

| O quê | Caminho |
|---|---|
| Notebook principal | `bertimbau_2/bertimbau_tuning.ipynb` |
| EDA interativa (localhost) | `eda_app/` (`python eda_app/app.py`) |
| Splits atuais | `bertimbau_2/{train,dev,test}_bert.csv` |
| Filtro de quase-duplicatas | `bertimbau_2/filtra_quase_duplicatas.py`, `bertimbau_2/saida_checagem/` (gerada antes da limpeza do ICL) |
| Comparação com outros modelos (futuro) | `PLANO_COMPARACAO_MODELOS.md` |
| Limpeza do rodapé ICL | `bertimbau_2/limpa_assinatura_icl.py` |
| Checagens (portal, truncamento, TF-IDF) | `bertimbau_2/checagem_leakage.py` |
| Auditoria / análise exploratória | `AUDITORIA_DATALEAKAGE.md`, `ANALISE_EXPLORATORIA.md` (seção 11: evolução dos dados) |
| Baseline TF-IDF+SVM (B1) | `/home/paulo/CascadeProjects/Applied_ML/NEW_training/training/results/tfidf/svm/experiment_b1_oversampling/` |
| Histórico dos experimentos de baseline | `.../NEW_training/new_chronology/PROGRESS.md`, `.../training/results/CONSOLIDACAO_FINAL.md` |
| Pré-processamento pesado (TF-IDF) | `.../NEW_training/FOR_TRAINING/`, `.../FOR_TEST/Pre_processed_for_Sparse` |
| Experimento de referência (viés) | `/home/paulo/CascadeProjects/Applied_ML/to_zip/NDMAIS_BIAS/` — `ndmais_doc/DOMAIN_BIAS_REPORT.md`, `ndmais_doc/CONSOLIDACAO_FINAL.md`, `ndmais_doc/PROGRESS.md`, `CROSS_PORTAL_TEST/PLANO.md`, `CROSS_PORTAL_TEST/results_all12/REPORT_ALL12.md`, `scripts/task8_domain_bias.py`, `CROSS_PORTAL_TEST/build_subsets.py` |
