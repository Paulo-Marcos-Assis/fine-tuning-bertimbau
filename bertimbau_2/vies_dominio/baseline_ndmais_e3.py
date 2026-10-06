#!/usr/bin/env python3
"""
baseline_ndmais_e3.py — Task T7 & T7b do PLANO_VIES_DOMINIO.md

Executa o experimento E3 para o modelo Baseline TF-IDF + LinearSVC:
1. Treina o modelo em 3 configurações de razão (natural 1:106, 1:64 e 1:10)
2. Avalia no dev in-domain do ndmais e seleciona a melhor razão via AP
3. Calibra o limiar ótimo de decisão no dev (threshold tuning)
4. Avalia a razão vencedora no:
   - Teste in-domain (test.csv do ndmais)
   - Conjunto de exposição interportais correspondente (exposicao_*.csv)
5. Calcula intervalos de confiança por bootstrap (2.000 iterações)
6. Salva as predições e gera o relatório comparativo formal
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ndmais_only")

print("=" * 75)
print("TASK T7 & T7b: Treinamento e Avaliação do Baseline E3 (TF-IDF + LinearSVC)")
print("=" * 75)

# Carrega dev e teste in-domain
print("\n[1/6] Carregando dev e teste in-domain (ndmais.com.br)...")
df_dev = pd.read_csv(os.path.join(DATA_DIR, "dev.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
y_dev = df_dev["label"].to_numpy()
y_test = df_test["label"].to_numpy()

print(f"  Dev in-domain:   {len(df_dev)} linhas ({y_dev.sum()} positivos, {len(y_dev)-y_dev.sum()} negativos)")
print(f"  Teste in-domain: {len(df_test)} linhas ({y_test.sum()} positivos, {len(y_test)-y_test.sum()} negativos)")

# Configurações de treino
configs = [
    {"name": "natural", "file": "train_natural.csv", "ratio_str": "1:106.4"},
    {"name": "1a64",    "file": "train_1a64.csv",    "ratio_str": "1:64.0"},
    {"name": "1a10",    "file": "train_1a10.csv",    "ratio_str": "1:10.0"},
]

def encontrar_melhor_limiar(scores, y_true):
    """Encontra o limiar no score do decision_function que maximiza o F1."""
    limiares = np.percentile(scores, np.linspace(0.1, 99.9, 1000))
    melhor_f1 = -1.0
    melhor_thr = 0.0
    for thr in limiares:
        preds = (scores >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, pos_label=1, average="binary", zero_division=0)
        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_thr = thr
    return melhor_thr, melhor_f1

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def calcular_metricas(y_true, scores, thr):
    preds = (scores >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, pos_label=1, average="binary", zero_division=0)
    ap = average_precision_score(y_true, scores)
    roc = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "ap": ap,
        "roc_auc": roc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pct_pred_pos": float(preds.mean() * 100)
    }

def bootstrap_ci(y_true, scores, thr, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aps, f1s, recs, precs = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        sc_b = scores[idx]
        pr_b = (sc_b >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_b, pr_b, pos_label=1, average="binary", zero_division=0)
        aps.append(average_precision_score(y_b, sc_b))
        f1s.append(f)
        recs.append(r)
        precs.append(p)
    return {
        "f1_ci": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
        "ap_ci": (float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))),
        "recall_ci": (float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))),
        "precision_ci": (float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5))),
    }

resultados_dev = {}
modelos_treinados = {}

# 2. Treinamento das 3 configurações e Avaliação no Dev
print("\n[2/6] Treinando o Baseline TF-IDF + LinearSVC nas 3 configurações...")
for cfg in configs:
    name = cfg["name"]
    print(f"\n--- Configuração: {name} ({cfg['ratio_str']}) ---")
    df_train = pd.read_csv(os.path.join(DATA_DIR, cfg["file"]))
    y_train = df_train["label"].to_numpy()
    
    vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=False,
        norm="l2",
        use_idf=True
    )
    X_train = vec.fit_transform(df_train["processed_text"])
    X_dev = vec.transform(df_dev["processed_text"])
    X_test = vec.transform(df_test["processed_text"])
    
    # LinearSVC com C=0.1 fixo (conforme config B1)
    clf = LinearSVC(C=0.1, random_state=SEED, max_iter=10000, dual="auto")
    clf.fit(X_train, y_train)
    
    scores_dev = clf.decision_function(X_dev)
    scores_test = clf.decision_function(X_test)
    
    # Métricas com limiar padrão 0.0
    m_padrao = calcular_metricas(y_dev, scores_dev, thr=0.0)
    
    # Encontra limiar ótimo no dev
    melhor_thr, f1_otimo = encontrar_melhor_limiar(scores_dev, y_dev)
    m_otimo = calcular_metricas(y_dev, scores_dev, thr=melhor_thr)
    
    print(f"  Dev AP (PR-AUC):            {m_padrao['ap']:.4f}")
    print(f"  Dev F1 (corte padrão 0.0):  {m_padrao['f1']:.4f} (Rec: {m_padrao['recall']:.4f}, Prec: {m_padrao['precision']:.4f})")
    print(f"  Dev F1 (corte ótimo {melhor_thr:.3f}): {m_otimo['f1']:.4f} (Rec: {m_otimo['recall']:.4f}, Prec: {m_otimo['precision']:.4f})")
    
    resultados_dev[name] = {
        "cfg": cfg,
        "vectorizer": vec,
        "clf": clf,
        "melhor_thr": melhor_thr,
        "m_padrao": m_padrao,
        "m_otimo": m_otimo,
        "scores_dev": scores_dev,
        "scores_test": scores_test
    }
    
    # Salva predições do dev
    df_dev_pred = pd.DataFrame({
        "url": df_dev["url"],
        "portal": df_dev["portal"],
        "label": y_dev,
        "decision_score": scores_dev,
        "p_fraude": sigmoid(scores_dev)
    })
    df_dev_pred.to_csv(os.path.join(DATA_DIR, f"pred_baseline_dev_{name}.csv"), index=False)

# 3. Escolha da Razão Vencedora no Dev (Task T7b)
print("\n[3/6] Escolhendo a melhor configuração no Dev (critério: AP)...")
# O critério primário estabelecido é o AP no dev
melhor_config_nome = max(resultados_dev.keys(), key=lambda k: resultados_dev[k]["m_padrao"]["ap"])
vencedor = resultados_dev[melhor_config_nome]
print(f"  ★ CONFIGURAÇÃO VENCEDORA NO DEV: '{melhor_config_nome}' com AP = {vencedor['m_padrao']['ap']:.4f} e F1 ótimo = {vencedor['m_otimo']['f1']:.4f}")
print(f"  Limiar calibrado no dev a ser aplicado: {vencedor['melhor_thr']:.4f}")

# 4. Avaliação no Teste In-Domain (ndmais)
print(f"\n[4/6] Avaliando a vencedora ('{melhor_config_nome}') no Teste In-Domain (ndmais)...")
scores_test_vencedor = vencedor["scores_test"]
thr_vencedor = vencedor["melhor_thr"]

m_test = calcular_metricas(y_test, scores_test_vencedor, thr=thr_vencedor)
ci_test = bootstrap_ci(y_test, scores_test_vencedor, thr=thr_vencedor)

print(f"  Teste In-Domain AP:        {m_test['ap']:.4f} [95% CI: {ci_test['ap_ci'][0]:.4f} - {ci_test['ap_ci'][1]:.4f}]")
print(f"  Teste In-Domain F1:        {m_test['f1']:.4f} [95% CI: {ci_test['f1_ci'][0]:.4f} - {ci_test['f1_ci'][1]:.4f}]")
print(f"  Teste In-Domain Recall:    {m_test['recall']:.4f} [95% CI: {ci_test['recall_ci'][0]:.4f} - {ci_test['recall_ci'][1]:.4f}]")
print(f"  Teste In-Domain Precision: {m_test['precision']:.4f}")
print(f"  Matriz de Confusão: TP={m_test['tp']}, FP={m_test['fp']}, FN={m_test['fn']}, TN={m_test['tn']}")

df_test_pred = pd.DataFrame({
    "url": df_test["url"],
    "portal": df_test["portal"],
    "label": y_test,
    "decision_score": scores_test_vencedor,
    "p_fraude": sigmoid(scores_test_vencedor)
})
df_test_pred.to_csv(os.path.join(DATA_DIR, f"pred_baseline_test_indomain_{melhor_config_nome}.csv"), index=False)

# 5. Avaliação na Exposição Interportais (Cross-Portal)
exp_file = f"exposicao_{melhor_config_nome}.csv" if melhor_config_nome != "natural" else "exposicao_1a106.csv"
print(f"\n[5/6] Avaliando a vencedora na Exposição Interportais ({exp_file})...")
df_exp = pd.read_csv(os.path.join(DATA_DIR, exp_file))
y_exp = df_exp["label"].to_numpy()

X_exp = vencedor["vectorizer"].transform(df_exp["processed_text"])
scores_exp = vencedor["clf"].decision_function(X_exp)

m_exp = calcular_metricas(y_exp, scores_exp, thr=thr_vencedor)
ci_exp = bootstrap_ci(y_exp, scores_exp, thr=thr_vencedor)

# Também calcula com o limiar ótimo da própria exposição (teto teórico para diagnóstico de shift)
thr_exp_otimo, f1_exp_teto = encontrar_melhor_limiar(scores_exp, y_exp)

print(f"  Exposição AP:              {m_exp['ap']:.4f} [95% CI: {ci_exp['ap_ci'][0]:.4f} - {ci_exp['ap_ci'][1]:.4f}]")
print(f"  Exposição F1 (limiar dev): {m_exp['f1']:.4f} [95% CI: {ci_exp['f1_ci'][0]:.4f} - {ci_exp['f1_ci'][1]:.4f}]")
print(f"  Exposição Recall:          {m_exp['recall']:.4f} [95% CI: {ci_exp['recall_ci'][0]:.4f} - {ci_exp['recall_ci'][1]:.4f}]")
print(f"  Exposição Precision:       {m_exp['precision']:.4f}")
print(f"  Exposição F1 (teto ót):    {f1_exp_teto:.4f} (com corte reajustado={thr_exp_otimo:.3f})")
print(f"  Matriz de Confusão: TP={m_exp['tp']}, FP={m_exp['fp']}, FN={m_exp['fn']}, TN={m_exp['tn']}")

df_exp_pred = pd.DataFrame({
    "url": df_exp["url"],
    "portal": df_exp["portal"],
    "label": y_exp,
    "decision_score": scores_exp,
    "p_fraude": sigmoid(scores_exp)
})
df_exp_pred.to_csv(os.path.join(DATA_DIR, f"pred_baseline_exposicao_{melhor_config_nome}.csv"), index=False)

# 6. Geração do Relatório Formal
print("\n[6/6] Consolidando relatório de resultados...")
relatorio_path = os.path.join(DATA_DIR, "RELATORIO_BASELINE_E3.md")
with open(relatorio_path, "w", encoding="utf-8") as f:
    f.write("# Relatório de Resultados — Baseline E3: TF-IDF + LinearSVC\n\n")
    f.write(f"**Data:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"**Arquitetura:** TF-IDF (n-gram 1-3, 20.000 features, l2) + `LinearSVC(C=0.1, random_state=42)`\n\n")
    
    f.write("## 1. Comparação das 3 Razões de Treino no Conjunto Dev In-Domain (ndmais)\n\n")
    f.write("| Configuração | Razão Treino | Dev AP (PR-AUC) | Dev F1 (Corte 0.0) | Limiar Dev Ótimo | Dev F1 (Limiar Dev) | Dev Recall | Dev Precision |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for name, res in resultados_dev.items():
        mp = res["m_padrao"]
        mo = res["m_otimo"]
        f.write(f"| `{name}` | {res['cfg']['ratio_str']} | **{mp['ap']:.4f}** | {mp['f1']:.4f} | {res['melhor_thr']:.3f} | **{mo['f1']:.4f}** | {mo['recall']:.4f} | {mo['precision']:.4f} |\n")
    
    f.write(f"\n> **Configuração Vencedora Selecionada:** `{melhor_config_nome}` (maior AP no dev: **{vencedor['m_padrao']['ap']:.4f}**). Limiar fixado: **{thr_vencedor:.4f}**.\n\n")
    
    f.write("## 2. Desempenho do Modelo Vencedor: In-Domain vs. Exposição Cross-Portal\n\n")
    f.write("| Cenário de Avaliação | Dataset | AP (PR-AUC) [95% CI] | F1 [95% CI] | Recall [95% CI] | Precision | % Previsto Fraude |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    f.write(f"| **In-Domain (ndmais)** | `test.csv` (n={len(df_test)}) | **{m_test['ap']:.4f}** [{ci_test['ap_ci'][0]:.4f} - {ci_test['ap_ci'][1]:.4f}] | **{m_test['f1']:.4f}** [{ci_test['f1_ci'][0]:.4f} - {ci_test['f1_ci'][1]:.4f}] | {m_test['recall']:.4f} [{ci_test['recall_ci'][0]:.4f} - {ci_test['recall_ci'][1]:.4f}] | {m_test['precision']:.4f} | {m_test['pct_pred_pos']:.2f}% |\n")
    f.write(f"| **Cross-Portal (Exposição)** | `{exp_file}` (n={len(df_exp)}) | **{m_exp['ap']:.4f}** [{ci_exp['ap_ci'][0]:.4f} - {ci_exp['ap_ci'][1]:.4f}] | **{m_exp['f1']:.4f}** [{ci_exp['f1_ci'][0]:.4f} - {ci_exp['f1_ci'][1]:.4f}] | {m_exp['recall']:.4f} [{ci_exp['recall_ci'][0]:.4f} - {ci_exp['recall_ci'][1]:.4f}] | {m_exp['precision']:.4f} | {m_exp['pct_pred_pos']:.2f}% |\n\n")
    
    f.write("## 3. Diagnóstico de Viés de Domínio (Shift de Limiar)\n\n")
    f.write(f"- **F1 com Limiar do Dev:** `{m_exp['f1']:.4f}`\n")
    f.write(f"- **F1 com Teto Ótimo na Exposição:** `{f1_exp_teto:.4f}` (corte reajustado para `{thr_exp_otimo:.3f}`)\n")
    f.write("- **Interpretação:** ")
    if m_exp['ap'] >= 0.70:
        f.write("O modelo mantém alto poder de ranqueamento (AP robusto) nos portais não vistos. A variação de F1 decorre primariamente do deslocamento na distribuição de scores (prior shift / threshold shift), e não da perda de capacidade discriminativa.\n")
    else:
        f.write("O modelo sofreu degradação substancial de ranqueamento ao ser exposto a fontes desconhecidas.\n")

print(f"\n[OK] Relatório completo gravado em: {relatorio_path}")
print("Execução da Task T7/T7b concluída com sucesso!")
