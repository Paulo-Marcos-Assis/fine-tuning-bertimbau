# Plano — Teste de viés de domínio (portal) no fine-tuning do BERTimbau

**Data:** 2026-09-30 · **Status:** PLANEJAMENTO — nada deste plano foi implementado ainda.
**Público-alvo:** agentes/colaboradores que vão implementar as tasks abaixo. Leia as seções 1–3 antes de qualquer task.

## 0. Regras de trabalho (valem para qualquer agente)

1. **Não fazer commit nem push sem autorização explícita do usuário.** O fluxo é: editar no clone local → o usuário faz push → `git pull` no Jupyter (https://jupyter.ceos.ufsc.br, GPUs RTX A6000) só nos momentos de execução pesada.
2. **O conjunto de teste (`bertimbau_2/test_bert.csv`, 11.374 linhas) não pode influenciar nenhuma decisão** (hiperparâmetro, limiar, checkpoint, escolha de modelo). Limiares e seleção saem só do dev.
3. **Não sobrescrever** `train/dev/test_bert.csv`. Artefatos novos vão em pastas novas (sugestão: `bertimbau_2/vies_dominio/`).
4. Antes de treinar na GPU, conferir `nvidia-smi` e ajustar `CUDA_VISIBLE_DEVICES` (em 2026-09-30 as GPUs 0 e 1 estavam livres; a 2 tinha 44 GB ocupados por outro usuário).
5. Documentar cada etapa como **evolução** (não substituir o registro anterior) em `AUDITORIA_DATALEAKAGE.md` / `ANALISE_EXPLORATORIA.md`. Texto objetivo — vai servir de base para o artigo.
6. Idioma: português.

## 1. Contexto

### 1.1 O repositório atual
- Fine-tuning de `neuralmind/bert-base-portuguese-cased` para detectar notícias de fraude/irregularidade (label 1) vs. normais (label 0). Notebook: `bertimbau_2/bertimbau_tuning.ipynb` (loss ponderada, seed 42, bf16, 3 épocas, seleção do checkpoint por F1 no dev, avaliação final única na PARTE 9).
- Dados (`bertimbau_2/*.csv`, colunas `url, processed_text, label, portal`), estado atual após as etapas de `ANALISE_EXPLORATORIA.md` seção 11:

| Split | Linhas | Positivos | % fraude |
|---|---|---|---|
| train | 36.650 | 562 | 1,533% |
| dev | 9.162 | 141 | 1,539% |
| test | 11.374 | 175 | 1,539% |

- Origem dos dados: pipeline em `/home/paulo/CascadeProjects/Applied_ML/NEW_training/` (mesmos splits; lá o teste tinha 11.454 linhas antes da deduplicação e do filtro de quase-duplicatas).

### 1.2 O problema: o portal está confundido com a classe
Dados medidos em `bertimbau_2/saida_checagem/portais_por_classe.csv` (teste original) e recontados nos CSVs atuais:

| Grupo | Linhas | Positivos | % fraude |
|---|---|---|---|
| `ndmais.com.br` | 45.427 | 410 | 0,9% |
| `nsctotal.com.br` | ~10.600 | 47 | 0,44% |
| **Todos os outros (27 domínios)** | 11.759 (inclui nsctotal) | **468** | — |

- Fora dos 2 maiores portais: 2,1% das linhas concentram **48% dos positivos**.
- `iclnoticias.com.br` (261 linhas, 100% fraude), `g1.globo.com` (62, 100%) e `jornalconexao.com.br` (76, 53%) somam 363 positivos (**41% de todos os positivos**). Portais institucionais (`mpsc`, `pc.sc.gov.br`, `tcesc`) são 100% fraude mas têm ~10 exemplos.
- **Baseline que usa só o portal (regressão logística, limiar no dev): F1 0,586 · AP 0,462** no teste (prevalência 0,015). Se o BERT não superar isso com folga, não há evidência de que aprendeu conteúdo.
- Indício de vazamento de "assinatura" da fonte no texto (contagem no `processed_text`): o termo "icl notícias" aparece em 33 textos, **100% positivos**; "jornal conexão" 4/5 positivos; "g1" 12,9% positivos. Vale testar mascaramento (Task 5).

### 1.3 Outros riscos já documentados (não são o foco deste plano, mas interagem)
- Truncamento em 512 tokens atinge ~74% dos positivos vs ~33% dos negativos (`saida_checagem/truncamento_por_classe.csv`).
- Quase-duplicatas (séries `parte-1/2`, agenda de candidatos, cinema, lua): removidas do teste só acima de 0,9 de cosseno TF-IDF; restam 161 pares entre 0,8 e 0,9.

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
3. **F1 por portal é enganoso:** portais 100% positivos dão F1 = 1 trivial; portais sem positivos dão F1 = 0. → Por portal reportar **recall nos positivos** e **taxa de falso positivo nos negativos**, nunca F1.
4. O pool cross-portal de lá era pequeno (199 pos, 59 portais). Aqui há **468 positivos fora do ndmais**, mas 78% vêm de só 3 portais — a conclusão "generaliza para outros portais" é, na prática, sobre essas 3 fontes.

## 3. Baseline de comparação (confirmação do F1 0,7420)

**Confirmado**, com ressalvas. Em `/home/paulo/CascadeProjects/Applied_ML/NEW_training/`:
- **Experimento B1 = TF-IDF + LinearSVC + random oversampling 1:10 (dentro de cada fold), sem `class_weight`**, TF-IDF n-grama (1,3), 20k features, `C=0.1`, CV 5-fold estratificado. Resultados em `training/results/tfidf/svm/experiment_b1_oversampling/` (`COMPARISON.md`, `experiment_b1_results.json`, `model.pkl`, script `experiment_b1_oversampling.py`; vetorizador em `experiment_a/vectorization/vectorizer.pkl`). Narrativa em `new_chronology/PROGRESS.md` (Task 6).
- Métricas **no dev**: F1 **0,7420**, precision 0,7394, recall 0,7447, PR-AUC 0,7882, ROC-AUC 0,9926. Supera o baseline da Task 5 (F1 0,7075, que é o "melhor" citado em `training/results/CONSOLIDACAO_FINAL.md`) e o undersampling B2 (0,7010).
- **Ressalvas:**
  - É F1 no **dev** (9.163 linhas, anterior à dedup de 1 linha), e a configuração foi escolhida comparando no dev. **Não há avaliação do B1 no teste** (a Task 7 do NEW_training não foi executada para ele).
  - O TF-IDF usa o texto do **pré-processamento pesado** (sem acento, minúsculas, sem stopwords; `NEW_training/FOR_TEST/Pre_processed_for_Sparse`, `vectorization/test_preprocessed.csv`), não o `processed_text` do BERT. A ligação entre os dois é a `url`.
  - Treinado no treino pré-dedup (36.652 linhas) e o teste de lá tem 11.454 linhas.

## 4. Decisões recomendadas (responde às dúvidas do usuário)

**4.1 Comparar com 1 baseline em vez de 12 modelos: sim.** O objetivo aqui é medir o fine-tuning, não repetir a grade de classificadores. Comparar: (a) BERTimbau fine-tuned e (b) **um** baseline = TF-IDF + LinearSVC (config B1), além do (c) baseline só-portal. Os 12 modelos do NDMAIS_BIAS não entram (o experimento de lá usa embeddings congelados + cabeça linear, que é outra família e colapsou no cross-portal).

**4.2 Quando fazer — em três níveis, por custo:**

| Nível | O que é | Quando | Custo |
|---|---|---|---|
| **E1** | Quebrar as métricas do modelo **já treinado** por portal (ndmais vs. fora do ndmais; recall/FPR por portal) | **Depois** da execução principal, sem retreinar | ~zero, mas exige salvar predições (ver abaixo) |
| **E2** | Reimplementar o baseline TF-IDF+SVM (B1) neste repo, com o mesmo teste | **Em paralelo agora** (CPU, local, minutos) | baixo |
| **E3** | Treinar só com `ndmais` (ambas as classes) e testar em outros portais — análogo direto do NDMAIS_BIAS — para o BERT e para o baseline | **Depois** da execução principal, como 2ª rodada na GPU | 1 fine-tuning extra (menor: ~28,9k linhas de treino ndmais) |
| E4 (opcional) | Mascarar assinaturas de portal no texto (ex.: "ICL Notícias") e reavaliar | Só se E1/E3 indicarem atalho | baixo |

**Única coisa a fazer antes de rodar o notebook principal:** acrescentar uma célula que **salva as predições** (probabilidade da classe fraude + `url` + `label`) de dev e teste em CSV. Sem isso, E1 exigiria retreinar. A execução principal não muda em mais nada e não depende do resultado de E1–E4.

## 5. Desenho dos experimentos

### E1 — Métricas por portal (sem retreino)
- Entrada: predições salvas (`url, portal, label, p_fraude`) do BERT e do baseline.
- Cortes: `ndmais` × `nsctotal` × "outros"; tabela por portal com n, positivos, **recall, FPR**, AP do grupo.
- Limiar: escolhido no **dev** (F1 máx.) e aplicado ao teste; reportar também AP (sem limiar).
- IC por bootstrap (2.000 reamostras) — positivos são poucos (teste: 175; fora do ndmais no teste: 94).

### E2 — Baseline TF-IDF + LinearSVC (config B1)
- Reimplementar o script B1 lendo os CSVs atuais (ou reaproveitar `vectorizer.pkl`/`model.pkl` e prever no teste filtrado por `url`). Hiperparâmetros fixos da config B1; não re-tunar no teste.
- Saída no **mesmo formato** de predições do BERT, para o E1 tratar os dois igual.

### E3 — Treino só-ndmais → teste cross-portal
- Treino/dev: linhas `ndmais` dos splits atuais (treino 28.985 linhas/270 pos; dev 7.351/59 pos). Teste in-domain: ndmais do teste (9.091/81 pos).
- **Pool cross-portal:** linhas não-ndmais dos três splits (11.759 linhas, 468 pos: 292 treino/82 dev/94 teste). Como o modelo não treinou com elas, pode-se usar todas — mas **(i)** remover as com similaridade TF-IDF ≥ 0,9 a qualquer linha do treino ndmais (mesmo critério da etapa 3) e **(ii)** o limiar vem só do dev-ndmais.
- Cuidado: os negativos não-ndmais são dominados por `nsctotal` (~10,6k); só ~800 negativos vêm de outros portais (bbc, gazeta, cartacapital…). FPR por portal, não agregado.
- Opcional (E3b, leave-one-portal-out): treinar sem `iclnoticias` (ou `g1`/`jornalconexao`) e testar nele — isola a dependência de cada fonte positiva.
- Interpretação: se o modelo ndmais-only (portal não confundido) mantém recall/AP razoáveis nos outros portais, o conteúdo generaliza; se o modelo completo for muito melhor só porque viu esses portais, isso **não** distingue atalho de simples cobertura — por isso o E3b.

### E4 — Mascaramento de assinatura (opcional)
- Remover do texto nomes/bordões de portal (ex.: "ICL Notícias", "Jornal Conexão", "Leia também", sufixos de portal) e reavaliar só o baseline (barato); depois o BERT, se relevante.
- Para o BERT não há atribuição por coeficiente; usar a variação entre com/sem máscara. Para o baseline TF-IDF, listar os top-20 coeficientes e marcar termos de portal (análogo à Task 8, mas com lista de termos de portal derivada dos dados, não só palavras-chave de fraude).

### Métricas (todas as tasks)
- Primária: **AP/PR-AUC**. Secundárias: precision/recall/F1 **em limiar escolhido no dev**; matriz de confusão; IC bootstrap.
- Por portal: recall (positivos) e FPR (negativos) — não F1 (ver 2.1.3).
- Sempre comparar contra o baseline só-portal (F1 0,586 · AP 0,462).
- Se possível, 3–5 seeds no fine-tuning (o notebook fixa só a seed 42).

## 6. Tasks

Cada task termina com critérios de aceite. Marcar como feito neste arquivo (seção 8).

| # | Task | Depende de | Entregável | Aceite |
|---|---|---|---|---|
| T1 | **Salvar predições no notebook:** após `trainer.predict`, gravar CSV com `url, portal, label, p_fraude` para dev e teste (`bertimbau_2/vies_dominio/pred_bert_{dev,test}.csv`). Incluir também o dev (para o limiar). | — | célula nova no notebook | Linhas = 9.162 e 11.374; `url` alinhado aos CSVs; nenhum uso do teste antes da PARTE 9 |
| T2 | **Baseline TF-IDF+SVM (B1)** reproduzido no repo atual, com predições no mesmo formato (`pred_tfidf_{dev,test}.csv`). Decidir se reaproveita `model.pkl` do NEW_training ou retreina (recomendado: retreinar com a config B1 nos CSVs atuais, script versionado). | — | `bertimbau_2/vies_dominio/baseline_tfidf_svm.py` | F1 dev próximo de 0,74 (registrar a diferença vs. 0,7420 e o motivo); teste usado uma única vez |
| T3 | **Baseline só-portal** com predições no mesmo formato (já existe a lógica em `checagem_leakage.py::baseline_portal`). | — | `pred_portal_{dev,test}.csv` | Reproduz F1 ≈ 0,586 / AP ≈ 0,462 (no teste filtrado pode variar um pouco; registrar) |
| T4 | **Script de análise por portal (E1):** lê as predições, calcula métricas agregadas e por portal (recall/FPR), limiar do dev, bootstrap, tabelas `.csv` + resumo `.md`. | T1–T3 | `bertimbau_2/vies_dominio/analise_portal.py` | Gera `metricas_por_portal.csv` e `resumo.md`; nenhuma métrica por portal em F1 |
| T5 | **Mascaramento (E4) no baseline TF-IDF:** top-20 coeficientes com marcação de termos de portal + reavaliação com texto mascarado. Lista de termos de portal derivada dos dados (termos com alta concentração em um portal). | T2 | `mascara_portal.py`, seção no resumo | Tabela com/sem máscara (AP, F1 no limiar do dev) |
| T6 | **Preparar dados E3:** gerar `train/dev/test` só-ndmais e o pool cross-portal (com filtro de similaridade ≥ 0,9 vs. treino ndmais); reportar contagens. Reaproveitar `build_subsets.py` do NDMAIS_BIAS só como referência (lá os subsets eram de tamanho fixo, aqui usar o pool inteiro). | — | `bertimbau_2/vies_dominio/ndmais_only/*.csv` + relatório de contagens | Contagens batem com a seção 5/E3; zero `url` em comum entre treino ndmais e pool |
| T7 | **Notebook E3 (BERT):** cópia parametrizada do notebook principal treinando só-ndmais (mesma config, seed 42; pesos de classe recalculados do novo treino) + predição in-domain e cross-portal, salvando no formato de T1. Rodar no Jupyter (conferir GPU livre). | T1, T6 | `bertimbau_ndmais_only.ipynb` | Treino concluído; predições salvas; nada do pool cross-portal usado para escolher checkpoint/limiar |
| T8 | **Baseline E3:** TF-IDF+SVM só-ndmais, mesmas saídas. | T2, T6 | predições | idem |
| T9 | **(Opcional) E3b leave-one-portal-out** para `iclnoticias`, `g1`, `jornalconexao`. | T7/T8 | tabela recall por portal retido | — |
| T10 | **Consolidar e documentar:** tabela final (modelo × {ndmais, cross-portal}, AP, F1@dev-thr, recall/FPR por portal, IC), conclusões e limitações; atualizar `AUDITORIA_DATALEAKAGE.md`/`ANALISE_EXPLORATORIA.md` como nova etapa. Texto curto (vai para o artigo). | T4 + (T5,T7–T9) | seção nova nos `.md` | Inclui a limitação "78% dos positivos não-ndmais vêm de 3 portais" |

**Ordem sugerida:** T1 (antes de rodar o notebook) → T2, T3 em paralelo, localmente → rodar o notebook principal no Jupyter → T4 → decidir se E3 vale o custo olhando o resultado de T4 → T6–T8 → T10. T5 e T9 só se necessário.

## 7. Riscos e pontos de atenção
- **Poucos positivos** → métricas instáveis; sempre IC e contagens absolutas. Diferenças de 2–3 pp de F1 são ruído.
- **Reuso de caminhos absolutos** dos scripts do NDMAIS_BIAS (`/home/paulo/CascadeProjects/Applied_ML/NDMAIS_BIAS`) — a pasta foi movida para `to_zip/`. Não copiar cegamente.
- **Texto do baseline ≠ texto do BERT** (pré-processamento pesado vs. leve): alinhar sempre pela `url`; conferir contagem de linhas após o join.
- **Escolha de limiar/configuração no teste** é o erro mais fácil de cometer aqui; o teste só para o resultado final de cada modelo.
- **Filtro de quase-duplicatas** foi aplicado só ao teste (≥ 0,9 vs. train+dev); no E3 o pool cross-portal precisa de filtro próprio contra o treino ndmais.
- O F1 perfeito do NDMAIS_BIAS in-domain (1,0) e "viés BAIXO" da Task 8 **não devem ser citados como evidência** de ausência de viés; o que vale dali é o cross-portal e a lição do limiar.

## 8. Checklist de progresso
- [x] T1 (células adicionadas ao notebook em 2026-09-30; falta executar no Jupyter) · [ ] T2 · [ ] T3 · [ ] T4 · [ ] T5 · [ ] T6 · [ ] T7 · [ ] T8 · [ ] T9 (opcional) · [ ] T10

## 9. Mapa de referências

| O quê | Caminho |
|---|---|
| Notebook principal | `bertimbau_2/bertimbau_tuning.ipynb` |
| Splits atuais | `bertimbau_2/{train,dev,test}_bert.csv` |
| Filtro de quase-duplicatas | `bertimbau_2/filtra_quase_duplicatas.py`, `bertimbau_2/saida_checagem/` |
| Checagens (portal, truncamento, TF-IDF) | `bertimbau_2/checagem_leakage.py` |
| Auditoria / análise exploratória | `AUDITORIA_DATALEAKAGE.md`, `ANALISE_EXPLORATORIA.md` (seção 11: evolução dos dados) |
| Baseline TF-IDF+SVM (B1) | `/home/paulo/CascadeProjects/Applied_ML/NEW_training/training/results/tfidf/svm/experiment_b1_oversampling/` |
| Histórico dos experimentos de baseline | `.../NEW_training/new_chronology/PROGRESS.md`, `.../training/results/CONSOLIDACAO_FINAL.md` |
| Pré-processamento pesado (TF-IDF) | `.../NEW_training/FOR_TRAINING/`, `.../FOR_TEST/Pre_processed_for_Sparse` |
| Experimento de referência (viés) | `/home/paulo/CascadeProjects/Applied_ML/to_zip/NDMAIS_BIAS/` — `ndmais_doc/DOMAIN_BIAS_REPORT.md`, `ndmais_doc/CONSOLIDACAO_FINAL.md`, `ndmais_doc/PROGRESS.md`, `CROSS_PORTAL_TEST/PLANO.md`, `CROSS_PORTAL_TEST/results_all12/REPORT_ALL12.md`, `scripts/task8_domain_bias.py`, `CROSS_PORTAL_TEST/build_subsets.py` |
