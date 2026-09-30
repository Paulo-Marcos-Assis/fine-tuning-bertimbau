#!/usr/bin/env python3
"""Remove de test_bert.csv as linhas cujo vizinho mais próximo (TF-IDF, cosseno) em
train+dev tem similaridade >= LIMIAR. Lê saida_checagem/quase_duplicatas_test.csv,
gerado por checagem_leakage.py. Idempotente (URLs já removidas são ignoradas)."""
import pandas as pd

LIMIAR = 0.9
dup = pd.read_csv("saida_checagem/quase_duplicatas_test.csv")
urls = set(dup.loc[dup["sim"] >= LIMIAR, "url"])
df = pd.read_csv("test_bert.csv")
manter = ~df["url"].isin(urls)
print(f"test: {len(df)} -> {manter.sum()} linhas (removidas {(~manter).sum()}, sim >= {LIMIAR})")
print("positivos:", int(df["label"].sum()), "->", int(df.loc[manter, "label"].sum()))
df[manter].to_csv("test_bert.csv", index=False)
