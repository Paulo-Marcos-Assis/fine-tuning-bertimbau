#!/usr/bin/env python3
"""
e5_probe.py — multilingual-e5-large como extrator congelado + LinearSVC.

Protocolo (comparável ao BERTimbau-geral e ao baseline T7c, sem E3/portal):
  full splits de bertimbau_2 (train 36650 / dev 9162 / test 11374),
  prefixo obrigatório "query: " (model card, linear probing),
  embeddings normalizados (1024-dim), GridSearchCV LinearSVC
  C=[0.1,0.5,1,2,5,10] x StratifiedKFold(5, seed 42), scoring=f1,
  métricas SOMENTE básicas a corte fixo 0.0 (accuracy/precision/recall/f1 + matriz).
Sem AP, sem bootstrap, sem threshold tuning.

Uso (na GPU): python3 e5_probe.py   # dentro de e5/
Requer: pip install sentence-transformers scikit-learn
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

SEED = 42
MODEL_ID = "intfloat/multilingual-e5-large"
PARAM_C = [0.1, 0.5, 1, 2, 5, 10]
ENCODE_BATCH = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "bertimbau_2"))


def metricas_basicas(y_true, y_pred):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(acc), "precision": float(prec),
        "recall": float(rec), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train_bert.csv"))
    df_dev = pd.read_csv(os.path.join(DATA_DIR, "dev_bert.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test_bert.csv"))
    for nome, df in [("treino", df_train), ("dev", df_dev), ("teste", df_test)]:
        assert {"url", "processed_text", "label", "portal"}.issubset(df.columns), nome
    y_train = df_train["label"].to_numpy()
    y_dev = df_dev["label"].to_numpy()
    y_test = df_test["label"].to_numpy()
    print(f"Treino: {len(df_train)} ({y_train.sum()} pos) | "
          f"Dev: {len(df_dev)} ({y_dev.sum()} pos) | "
          f"Teste: {len(df_test)} ({y_test.sum()} pos)")

    print(f"\nCarregando {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID)

    X, ys, dfs = {}, {}, {}
    for nome, df, y in [("train", df_train, y_train),
                        ("dev", df_dev, y_dev),
                        ("test", df_test, y_test)]:
        textos = ["query: " + t for t in df["processed_text"].tolist()]
        print(f"Codificando {nome} ({len(textos)} textos)...")
        X[nome] = model.encode(textos, batch_size=ENCODE_BATCH,
                               normalize_embeddings=True, show_progress_bar=True)
        ys[nome], dfs[nome] = y, df
    print("Dimensão:", X["train"].shape)

    grid = GridSearchCV(
        LinearSVC(max_iter=10000, dual="auto", random_state=SEED),
        param_grid={"C": PARAM_C},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="f1", refit=True, n_jobs=-1,
    )
    grid.fit(X["train"], ys["train"])
    print(f"\nBest C={grid.best_params_['C']} cv_f1={grid.best_score_:.4f}")
    clf = grid.best_estimator_

    linhas = []
    for nome in ["dev", "test"]:
        pred = clf.predict(X[nome])  # corte fixo 0.0
        m = metricas_basicas(ys[nome], pred)
        print(f"{nome}: f1={m['f1']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} acc={m['accuracy']:.4f} {m}")
        linhas.append({"split": nome, "best_C": grid.best_params_["C"], **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
        df = dfs[nome]
        pd.DataFrame({
            "url": df["url"].to_numpy(), "portal": df["portal"].to_numpy(),
            "label": ys[nome], "pred": pred,
            "decision_score": clf.decision_function(X[nome]),
        }).to_csv(os.path.join(BASE_DIR, f"pred_e5_{nome}.csv"), index=False)

    with open(os.path.join(BASE_DIR, "RESULTADOS_E5_PROBE.md"), "w", encoding="utf-8") as f:
        f.write("# e5-large probe — embeddings congelados + LinearSVC (corte 0.0)\n\n")
        f.write(f"Modelo: {MODEL_ID} | Best C={grid.best_params_['C']} "
                f"(cv_f1={grid.best_score_:.4f})\n\n")
        f.write("| split | N | pos | accuracy | precision | recall | f1 | tp | fp | fn | tn |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for nome, df, m in [("dev", df_dev, metricas_basicas(ys["dev"], clf.predict(X["dev"]))),
                            ("test", df_test, metricas_basicas(ys["test"], clf.predict(X["test"])))]:
            f.write(f"| {nome} | {len(df)} | {int(ys[nome].sum())} | {m['accuracy']:.4f} | "
                    f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                    f"{m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |\n")
    print("\nSalvos: pred_e5_dev.csv, pred_e5_test.csv, RESULTADOS_E5_PROBE.md")


if __name__ == "__main__":
    main()
