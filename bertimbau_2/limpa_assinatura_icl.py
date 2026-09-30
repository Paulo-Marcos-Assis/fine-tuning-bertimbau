#!/usr/bin/env python3
"""Remove o rodapé de assinatura do ICL Notícias ("Redação ICL Economia... Com informações da/do ...")
dos textos de train/dev/test_bert.csv (qualquer portal), sobrescrevendo os arquivos.

O crédito vai do "Redação ICL" até o fim do texto; se houver a seção "Relacionados" depois dele,
ela é mantida. Idempotente (rodar de novo não altera nada). Originais: histórico do git.
"""
import re

import pandas as pd

RODAPE_ICL = re.compile(r"\s*Redação ICL\s*\w*?\s*Com informações\b.*?(?=\s*Relacionados\b|\Z)", re.DOTALL)

total = 0
for s in ["train", "dev", "test"]:
    path = f"{s}_bert.csv"
    df = pd.read_csv(path)
    novo = df["processed_text"].map(lambda t: RODAPE_ICL.sub("", t).rstrip() if isinstance(t, str) else t)
    mudou = novo != df["processed_text"]
    print(f"{s}: {int(mudou.sum())} textos alterados | portais: {df.loc[mudou, 'portal'].value_counts().to_dict()}")
    total += int(mudou.sum())
    if mudou.any():
        df["processed_text"] = novo
        df.to_csv(path, index=False)
print("total alterado:", total)
