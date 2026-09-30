#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checagem_leakage.py — checagens complementares à auditoria de data leakage.

Lê train_bert.csv, dev_bert.csv e test_bert.csv (colunas: url, processed_text,
label, portal) e faz:

  1. URLs canônicas: pega variações de barra final, query string (?utm_...),
     "www." e sufixo "-2" que escapam da interseção exata de URL.
  2. Portais x classe: taxa de fraude por portal, concentração dos positivos,
     portais que só aparecem em alguns splits, páginas que não são notícia
     e um baseline que usa SÓ o portal (detecta atalho de fonte).
  3. Textos: comprimento por classe e os textos mais curtos de cada split.
  3b. Truncamento: mede quantos textos são cortados em 512 tokens
      (bert-base-portuguese-cased) por split e por classe.
  4. Quase-duplicatas: vizinho mais próximo por TF-IDF/cosseno
     (dev -> train; test -> train+dev).

Uso:
    python checagem_leakage.py --sets caminho/para/finetuning/sets
    python checagem_leakage.py --sets caminho/para/sets --sample 3000   # teste rápido

Saídas: tabelas no terminal + CSVs em --out (padrão: saida_checagem/).
Dependências: pandas, numpy, scipy, scikit-learn.
"""
import argparse
import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder

SPLITS = ["train", "dev", "test"]
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 90)


def secao(titulo):
    print("\n" + "=" * 80 + f"\n{titulo}\n" + "=" * 80)


def carrega(sets_dir, sample=0):
    dfs = {}
    for s in SPLITS:
        caminho = Path(sets_dir) / f"{s}_bert.csv"
        if not caminho.exists():
            raise SystemExit(f"Arquivo não encontrado: {caminho}")
        df = pd.read_csv(caminho, dtype={"url": "string", "processed_text": "string", "portal": "string"})
        df["url"] = df["url"].fillna("")
        df["processed_text"] = df["processed_text"].fillna("")
        df["portal"] = df["portal"].fillna("(vazio)")
        df["label"] = df["label"].astype(int)
        df["split"] = s
        if sample and len(df) > sample:
            df = df.sample(n=sample, random_state=0)
        dfs[s] = df.reset_index(drop=True)
    return dfs


# ---------------------------------------------------------------- 1. URLs
def canon_url(u, remove_sufixo=False):
    p = urlsplit(u.strip())
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/").lower()
    if remove_sufixo:
        path = re.sub(r"-\d{1,2}$", "", path)  # ex.: ...-saude-2 -> ...-saude
    return host + path


def checa_urls(all_df, out):
    secao("1. URLs canônicas (ignora barra final, query string, www e sufixo -N)")
    print("Atenção: se algum site usa query string como ID do artigo (?id=123),\n"
          "a canonicalização junta artigos diferentes. Confira os pares antes de concluir.\n")
    for nome, suf in [("basica", False), ("agressiva", True)]:
        df = all_df.copy()
        df["canon"] = df["url"].map(lambda u: canon_url(u, suf))
        tam = df.groupby("canon")["url"].transform("size")
        multi = df[tam > 1]
        if multi.empty:
            print(f"[{nome}] nenhum grupo com mais de uma URL.")
            continue
        grp = multi.groupby("canon").agg(
            n=("url", "size"),
            splits=("split", lambda s: "+".join(sorted(set(s)))),
            rotulos=("label", lambda s: sorted(set(int(x) for x in s))),
            urls=("url", lambda s: " | ".join(s)),
        )
        cross = grp[grp["splits"].str.contains(r"\+")]
        intra = grp[~grp["splits"].str.contains(r"\+")]
        conflito = grp[grp["rotulos"].map(len) > 1]
        print(f"[{nome}] grupos duplicados: {len(grp)} | entre splits: {len(cross)} | "
              f"dentro de um split: {len(intra)} | com rótulos conflitantes: {len(conflito)}")
        if len(cross):
            print(cross["splits"].value_counts().to_string())
        grp.to_csv(Path(out) / f"urls_canonicas_{nome}.csv", encoding="utf-8")
    print(f"\nCSVs em {out}/urls_canonicas_*.csv")


# ---------------------------------------------------------------- 2. Portais
def melhor_limiar(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def baseline_portal(dfs):
    enc = OneHotEncoder(handle_unknown="ignore")
    Xtr = enc.fit_transform(dfs["train"][["portal"]])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(Xtr, dfs["train"]["label"])
    p_dev = clf.predict_proba(enc.transform(dfs["dev"][["portal"]]))[:, 1]
    p_test = clf.predict_proba(enc.transform(dfs["test"][["portal"]]))[:, 1]
    thr = melhor_limiar(dfs["dev"]["label"], p_dev)  # limiar escolhido no dev, nunca no teste
    y = dfs["test"]["label"]
    pred = (p_test >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    ap = average_precision_score(y, p_test)
    print("\nBaseline que usa SÓ o portal (regressão logística, limiar escolhido no dev):")
    print(f"  teste -> precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  "
          f"AP={ap:.3f}  (prevalência = AP de um classificador aleatório = {y.mean():.3f})")
    print("  Leitura: quanto mais próximo esse F1/AP estiver do obtido pelo BERT, mais suspeito\n"
          "  é que o modelo esteja usando a fonte (portal) e não o conteúdo.")


def eh_pagina_suspeita(url):
    p = urlsplit(url.strip())
    host = p.netloc.lower()
    return (host.startswith("assinaturas.") or host.startswith("conta.") or host == "bsky.app"
            or p.path in ("", "/"))


def checa_portais(all_df, dfs, out):
    secao("2. Portais x classe")
    n = pd.crosstab(all_df["portal"], all_df["split"]).reindex(columns=SPLITS, fill_value=0)
    pos = (all_df.pivot_table(index="portal", columns="split", values="label", aggfunc="sum", fill_value=0)
           .reindex(index=n.index, columns=SPLITS, fill_value=0))
    tab = pd.DataFrame({"n_total": n.sum(axis=1), "pos_total": pos.sum(axis=1)})
    tab["%fraude"] = (100 * tab["pos_total"] / tab["n_total"]).round(2)
    for s in SPLITS:
        tab[f"n_{s}"] = n[s]
        tab[f"pos_{s}"] = pos[s]
    tab = tab.sort_values("n_total", ascending=False)
    print(tab.to_string())
    tab.to_csv(Path(out) / "portais_por_classe.csv", encoding="utf-8")

    tot_n, tot_pos = tab["n_total"].sum(), tab["pos_total"].sum()
    top2 = tab.index[:2]
    fora = tab.loc[~tab.index.isin(top2)]
    print(f"\nFora dos 2 maiores portais: {int(fora['n_total'].sum())} linhas "
          f"({100 * fora['n_total'].sum() / tot_n:.1f}% do total) e "
          f"{int(fora['pos_total'].sum())} positivos ({100 * fora['pos_total'].sum() / max(tot_pos, 1):.1f}% dos positivos).")
    conc = tab[(tab["n_total"] >= 5) & (tab["%fraude"] >= 10)]
    if len(conc):
        print("\nPortais com >=10% de fraude (n>=5) — candidatos a atalho:")
        print(conc[["n_total", "pos_total", "%fraude"]].to_string())

    so_fora_treino = tab[(tab["n_train"] == 0) & ((tab["n_dev"] > 0) | (tab["n_test"] > 0))]
    if len(so_fora_treino):
        print("\nPortais ausentes do treino (aparecem só em dev/test):")
        print(so_fora_treino[["n_dev", "pos_dev", "n_test", "pos_test"]].to_string())

    susp = all_df[all_df["url"].map(eh_pagina_suspeita)]
    print(f"\nPáginas possivelmente não-notícia (assinaturas./conta./bsky.app/home sem path): {len(susp)}")
    if len(susp):
        print(pd.crosstab(susp["portal"], [susp["split"], susp["label"]]).to_string())
        susp[["split", "label", "portal", "url"]].to_csv(Path(out) / "paginas_suspeitas.csv",
                                                          index=False, encoding="utf-8")
    baseline_portal(dfs)


# ---------------------------------------------------------------- 3b. Truncamento
def checa_truncamento(all_df, out, model="neuralmind/bert-base-portuguese-cased", max_len=512):
    secao(f"3b. Truncamento em {max_len} tokens ({model})")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as e:
        print(f"  Tokenizador não disponível ({e}) — pulando medição de tokens.")
        print("  Estimativa 4 chars/token já validada: mediana ~385 tokens, q75 ~610 -> ~33% cortados.")
        return
    all_df = all_df.copy()
    # tokeniza em blocos para não estourar memória
    textos = all_df["processed_text"].tolist()
    lens = []
    bs = 500
    for i in range(0, len(textos), bs):
        enc = tok(textos[i:i+bs], truncation=False, padding=False, add_special_tokens=True)
        lens.extend(len(x) for x in enc["input_ids"])
        print(f"    {min(i+bs, len(textos))}/{len(textos)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    all_df["tokens"] = lens
    all_df["truncado"] = all_df["tokens"] > max_len
    res = all_df.groupby(["split", "label"])["tokens"].describe(percentiles=[0.5, 0.75, 0.9, 0.95])[["count","min","50%","75%","90%","95%","max"]]
    print(res.round(0).to_string())
    g = all_df.groupby(["split", "label"])["truncado"].agg(["sum","count"])
    g["%"] = (100*g["sum"]/g["count"]).round(1)
    print("\nTruncados (>512):")
    print(g.to_string())
    # geral
    tot = all_df["truncado"].sum()
    print(f"\nGeral: {int(tot)}/{len(all_df)} = {100*tot/len(all_df):.1f}% cortados em {max_len}")
    print("Leitura: se positivos são mais longos, o truncamento atinge as classes de forma diferente.")
    print("Se indício de fraude está no fim da matéria, o modelo não o vê.")
    # salva
    tab = all_df.groupby(["split","label"]).agg(
        n=("tokens","count"), med=("tokens","median"), q75=("tokens", lambda s: s.quantile(0.75)),
        truncados=("truncado","sum")
    )
    tab["%trunc"] = (100*tab["truncados"]/tab["n"]).round(1)
    tab.to_csv(Path(out) / "truncamento_por_classe.csv", encoding="utf-8")
    print(f"\nCSV em {out}/truncamento_por_classe.csv")


# ---------------------------------------------------------------- 3. Textos
def checa_textos(all_df, out):
    secao("3. Textos: comprimento por classe e os mais curtos")
    all_df = all_df.copy()
    all_df["chars"] = all_df["processed_text"].str.len()
    res = all_df.groupby(["split", "label"])["chars"].describe(percentiles=[0.5, 0.9])[
        ["count", "min", "50%", "90%", "max"]]
    print(res.round(0).to_string())
    print("\nSe a mediana de comprimento difere muito entre as classes, o tamanho já é um sinal\n"
          "(e o truncamento em 512 tokens atinge as classes de forma diferente).")
    curtos = []
    for s in SPLITS:
        c = all_df[all_df["split"] == s].nsmallest(8, "chars")
        curtos.append(c)
        print(f"\n8 textos mais curtos de {s}:")
        for _, r in c.iterrows():
            print(f"  {int(r['chars']):>5} chars | label={r['label']} | {r['portal']} | "
                  f"{r['processed_text'][:70]!r}")
    pd.concat(curtos)[["split", "chars", "label", "portal", "url", "processed_text"]].to_csv(
        Path(out) / "textos_mais_curtos.csv", index=False, encoding="utf-8")


# ---------------------------------------------------------------- 4. Quase-duplicatas
def top1(Xq, Xr, chunk):
    """Vizinho mais próximo (cosseno; os vetores TF-IDF já são L2-normalizados)."""
    n = Xq.shape[0]
    sim = np.zeros(n, dtype=np.float32)
    idx = np.zeros(n, dtype=np.int64)
    XrT = Xr.T.tocsr()
    for i in range(0, n, chunk):
        S = (Xq[i:i + chunk] @ XrT).toarray()
        j = S.argmax(axis=1)
        idx[i:i + chunk] = j
        sim[i:i + chunk] = S[np.arange(len(j)), j]
        print(f"    {min(i + chunk, n)}/{n}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return sim, idx


def checa_quase_duplicatas(all_df, out, thr, min_sim, chunk):
    secao("4. Quase-duplicatas por TF-IDF (vizinho mais próximo)")
    print("Vetorizando (TF-IDF ajustado nos 3 splits; é não supervisionado, serve só para a auditoria)...")
    vec = TfidfVectorizer(min_df=2, max_df=0.5, max_features=200_000, sublinear_tf=True,
                          dtype=np.float32)
    X = vec.fit_transform(all_df["processed_text"])
    idx = {s: np.where(all_df["split"].values == s)[0] for s in SPLITS}
    consultas = {"dev": ["train"], "test": ["train", "dev"]}
    resumo = []
    for q, refs in consultas.items():
        print(f"\n  {q} -> {'+'.join(refs)}")
        q_idx = idx[q]
        r_idx = np.concatenate([idx[r] for r in refs])
        sim, j = top1(X[q_idx], X[r_idx], chunk)
        qd = all_df.iloc[q_idx].reset_index(drop=True)
        nn = all_df.iloc[r_idx[j]].reset_index(drop=True)
        res = pd.DataFrame({
            "sim": sim.round(4),
            "label": qd["label"], "nn_label": nn["label"],
            "portal": qd["portal"], "nn_portal": nn["portal"],
            "nn_split": nn["split"],
            "url": qd["url"], "nn_url": nn["url"],
            "texto": qd["processed_text"].str.slice(0, 200),
            "nn_texto": nn["processed_text"].str.slice(0, 200),
        })
        pos = res["label"] == 1
        print(f"  similaridade máxima — mediana: positivos={res.loc[pos, 'sim'].median():.3f} | "
              f"negativos={res.loc[~pos, 'sim'].median():.3f}")
        for t in (0.8, 0.9, 0.95, 0.99):
            m = res["sim"] >= t
            resumo.append({
                "consulta": q, "referencia": "+".join(refs), "limiar": t, "pares": int(m.sum()),
                "positivos_na_consulta": int((m & pos).sum()),
                "rotulos_diferentes": int((m & (res["label"] != res["nn_label"])).sum()),
                "portais_diferentes": int((m & (res["portal"] != res["nn_portal"])).sum()),
            })
        res[res["sim"] >= min_sim].sort_values("sim", ascending=False).to_csv(
            Path(out) / f"quase_duplicatas_{q}.csv", index=False, encoding="utf-8")
        topo = res[res["sim"] >= thr].sort_values("sim", ascending=False).head(10)
        if len(topo):
            print(f"\n  Top pares com sim >= {thr}:")
            print(topo[["sim", "label", "nn_label", "nn_split", "url", "nn_url"]].to_string(index=False))
    print("\nResumo (pares = linhas da consulta cujo vizinho tem similaridade >= limiar):")
    print(pd.DataFrame(resumo).to_string(index=False))
    print(f"\nCSVs em {out}/quase_duplicatas_*.csv (sim >= {min_sim}). Leia à mão os pares de topo.\n"
          "Calibração (teste com dados sintéticos): trocar ~5% das palavras de um texto já derruba\n"
          "a similaridade para ~0.80-0.86, então não use 0.9 como corte único. Como referência,\n"
          "pares não relacionados ficam perto de 0.1-0.2. Textos muito semelhantes de portais\n"
          "diferentes ('portais_diferentes' alto) sugerem o mesmo fato noticiado por vários veículos\n"
          "ou páginas geradas por um mesmo modelo.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Checagens complementares de data leakage.")
    ap.add_argument("--sets", required=True, help="pasta com train_bert.csv, dev_bert.csv, test_bert.csv")
    ap.add_argument("--out", default="saida_checagem", help="pasta de saída dos CSVs")
    ap.add_argument("--thr", type=float, default=0.8, help="limiar para listar os pares no terminal")
    ap.add_argument("--min-sim", type=float, default=0.7, help="similaridade mínima salva nos CSVs")
    ap.add_argument("--sample", type=int, default=0,
                    help="usa só N linhas aleatórias por split (teste rápido; quebra pares duplicados)")
    ap.add_argument("--chunk", type=int, default=400, help="linhas por bloco no cálculo de similaridade")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dfs = carrega(args.sets, args.sample)
    all_df = pd.concat([dfs[s] for s in SPLITS], ignore_index=True)
    print({s: len(dfs[s]) for s in SPLITS})

    checa_urls(all_df, out)
    checa_portais(all_df, dfs, out)
    checa_textos(all_df, out)
    checa_truncamento(all_df, out)
    checa_quase_duplicatas(all_df, out, args.thr, args.min_sim, args.chunk)


if __name__ == "__main__":
    main()