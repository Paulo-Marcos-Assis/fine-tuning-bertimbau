#!/usr/bin/env python3
"""
baseline_maxfeat_grid_e3.py — viés de domínio E3, versão simplificada.

Grade: max_features [1000, 3000, 5000, 10000] x razoes [natural, 1a64, 1a10] = 12 combos.
Modelo: TF-IDF(1-3gram, min_df=2, max_df=0.9, l2) + GridSearchCV LinearSVC
  C=[0.1, 0.5, 1, 2, 5, 10], StratifiedKFold(5, shuffle, seed 42), scoring=f1, refit.
Metricas SOMENTE basicas a corte fixo 0.0: accuracy, precision, recall, f1 + TP/FP/FN/TN.
Sem AP, sem bootstrap, sem threshold tuning.
Selecao em 2 niveis no dev: melhor k por razao (maior F1), depois razao vencedora
(maior F1). Avalia a vencedora 1x no teste in-domain + exposicao da mesma razao.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ndmais_only")

MAX_FEATURES = [1000, 3000, 5000, 10000]
PARAM_C = [0.1, 0.5, 1, 2, 5, 10]

CONFIGS = [
    {"name": "natural", "train": "train_natural.csv", "expo": "exposicao_1a106.csv"},
    {"name": "1a64", "train": "train_1a64.csv", "expo": "exposicao_1a64.csv"},
    {"name": "1a10", "train": "train_1a10.csv", "expo": "exposicao_1a10.csv"},
]


def metricas_basicas(y_true, y_pred):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    df_dev = pd.read_csv(os.path.join(DATA_DIR, "dev.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y_dev = df_dev["label"].to_numpy()
    y_test = df_test["label"].to_numpy()
    print(f"Dev: {len(df_dev)} linhas ({y_dev.sum()} pos)")
    print(f"Teste in-domain: {len(df_test)} linhas ({y_test.sum()} pos)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    linhas_dev = []

    for cfg in CONFIGS:
        df_train = pd.read_csv(os.path.join(DATA_DIR, cfg["train"]))
        y_train = df_train["label"].to_numpy()
        print(f"\n=== razao {cfg['name']} ({len(df_train)} linhas, {y_train.sum()} pos) ===")

        for k in MAX_FEATURES:
            vec = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=k,
                min_df=2,
                max_df=0.9,
                sublinear_tf=False,
                norm="l2",
                use_idf=True,
            )
            X_train = vec.fit_transform(df_train["processed_text"])
            X_dev = vec.transform(df_dev["processed_text"])

            grid = GridSearchCV(
                LinearSVC(max_iter=10000, dual="auto", random_state=SEED),
                param_grid={"C": PARAM_C},
                cv=cv,
                scoring="f1",
                refit=True,
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            best_c = grid.best_params_["C"]
            cv_f1 = float(grid.best_score_)

            clf = grid.best_estimator_
            y_pred_dev = clf.predict(X_dev)  # corte fixo 0.0 do LinearSVC
            m = metricas_basicas(y_dev, y_pred_dev)
            print(
                f"  k={k}: best_C={best_c} cv_f1={cv_f1:.4f} | "
                f"dev f1={m['f1']:.4f} prec={m['precision']:.4f} "
                f"rec={m['recall']:.4f} acc={m['accuracy']:.4f} "
                f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})"
            )
            linhas_dev.append({
                "razao": cfg["name"], "max_features": k, "best_C": best_c,
                "cv_f1": round(cv_f1, 4),
                "dev_accuracy": round(m["accuracy"], 4),
                "dev_precision": round(m["precision"], 4),
                "dev_recall": round(m["recall"], 4),
                "dev_f1": round(m["f1"], 4),
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            })

            pd.DataFrame({
                "url": df_dev["url"].to_numpy(),
                "portal": df_dev["portal"].to_numpy(),
                "label": y_dev,
                "pred": y_pred_dev,
                "decision_score": clf.decision_function(X_dev),
            }).to_csv(os.path.join(DATA_DIR, f"pred_baseline_dev_{cfg['name']}_k{k}.csv"), index=False)

    df_res = pd.DataFrame(linhas_dev).sort_values(["razao", "max_features"])
    df_res.to_csv(os.path.join(DATA_DIR, "RESULTADOS_MAXFEAT_GRID_DEV.csv"), index=False)
    print("\n--- Tabela dev (12 combos) ---")
    print(df_res.to_string(index=False))

    # Nivel 1: melhor k por razao (maior dev_f1; desempate: menor k)
    melhor_k = {}
    for razao in df_res["razao"].unique():
        sub = df_res[df_res["razao"] == razao].sort_values(["dev_f1", "max_features"], ascending=[False, True])
        melhor_k[razao] = int(sub.iloc[0]["max_features"])
    print("\nMelhor k por razao:", melhor_k)

    # Nivel 2: razao vencedora (maior dev_f1 entre as melhores de cada razao)
    cand = df_res[df_res.apply(lambda r: r["max_features"] == melhor_k[r["razao"]], axis=1)]
    win = cand.sort_values("dev_f1", ascending=False).iloc[0]
    razao_win, k_win = win["razao"], int(win["max_features"])
    print(f"VENCEDORA: razao={razao_win} k={k_win} dev_f1={win['dev_f1']:.4f} (C={win['best_C']})")

    # Re-treina a vencedora (mesmo protocolo) e avalia 1x no teste + exposicao
    cfg_win = next(c for c in CONFIGS if c["name"] == razao_win)
    df_train = pd.read_csv(os.path.join(DATA_DIR, cfg_win["train"]))
    y_train = df_train["label"].to_numpy()
    vec = TfidfVectorizer(
        ngram_range=(1, 3), max_features=k_win, min_df=2, max_df=0.9,
        sublinear_tf=False, norm="l2", use_idf=True,
    )
    X_train = vec.fit_transform(df_train["processed_text"])
    grid = GridSearchCV(
        LinearSVC(max_iter=10000, dual="auto", random_state=SEED),
        param_grid={"C": PARAM_C},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="f1", refit=True, n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_

    X_test = vec.transform(df_test["processed_text"])
    y_pred_test = clf.predict(X_test)
    m_test = metricas_basicas(y_test, y_pred_test)
    print(f"\nTeste in-domain: f1={m_test['f1']:.4f} prec={m_test['precision']:.4f} "
          f"rec={m_test['recall']:.4f} acc={m_test['accuracy']:.4f} {m_test}")
    pd.DataFrame({
        "url": df_test["url"].to_numpy(), "portal": df_test["portal"].to_numpy(),
        "label": y_test, "pred": y_pred_test,
        "decision_score": clf.decision_function(X_test),
    }).to_csv(os.path.join(DATA_DIR, f"pred_baseline_test_indomain_{razao_win}_k{k_win}.csv"), index=False)

    df_exp = pd.read_csv(os.path.join(DATA_DIR, cfg_win["expo"]))
    y_exp = df_exp["label"].to_numpy()
    y_pred_exp = clf.predict(vec.transform(df_exp["processed_text"]))
    m_exp = metricas_basicas(y_exp, y_pred_exp)
    print(f"Exposicao ({cfg_win['expo']}): f1={m_exp['f1']:.4f} prec={m_exp['precision']:.4f} "
          f"rec={m_exp['recall']:.4f} acc={m_exp['accuracy']:.4f} {m_exp}")
    pd.DataFrame({
        "url": df_exp["url"].to_numpy(), "portal": df_exp["portal"].to_numpy(),
        "label": y_exp, "pred": y_pred_exp,
        "decision_score": clf.decision_function(vec.transform(df_exp["processed_text"])),
    }).to_csv(os.path.join(DATA_DIR, f"pred_baseline_exposicao_{razao_win}_k{k_win}.csv"), index=False)

    with open(os.path.join(DATA_DIR, "RELATORIO_MAXFEAT_GRID_E3.md"), "w", encoding="utf-8") as f:
        f.write("# Baseline E3 simplificado — grade max_features (só métricas básicas, corte 0.0)\n\n")
        f.write(f"Vencedora: razao={razao_win} k={k_win} C={grid.best_params_['C']}\n\n")
        f.write("## Dev (12 combos)\n\n")
        cols = list(df_res.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, r in df_res.iterrows():
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
        f.write("## Vencedora — teste in-domain vs exposição\n\n")
        f.write(f"| cenário | dataset | N | pos | accuracy | precision | recall | f1 | tp | fp | fn | tn |\n")
        f.write(f"|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        f.write(f"| in-domain | test.csv | {len(y_test)} | {int(y_test.sum())} | {m_test['accuracy']:.4f} | {m_test['precision']:.4f} | {m_test['recall']:.4f} | {m_test['f1']:.4f} | {m_test['tp']} | {m_test['fp']} | {m_test['fn']} | {m_test['tn']} |\n")
        f.write(f"| exposição | {cfg_win['expo']} | {len(y_exp)} | {int(y_exp.sum())} | {m_exp['accuracy']:.4f} | {m_exp['precision']:.4f} | {m_exp['recall']:.4f} | {m_exp['f1']:.4f} | {m_exp['tp']} | {m_exp['fp']} | {m_exp['fn']} | {m_exp['tn']} |\n")
    print("Relatorio: RELATORIO_MAXFEAT_GRID_E3.md + RESULTADOS_MAXFEAT_GRID_DEV.csv")


if __name__ == "__main__":
    main()
