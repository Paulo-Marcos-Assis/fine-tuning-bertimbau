#!/usr/bin/env python3
"""EDA interativa dos splits (train/dev/test_bert.csv) em localhost.

Uso (da raiz do repositório):
    python eda_app/app.py                 # http://127.0.0.1:8050
    python eda_app/app.py --port 8060 --data bertimbau_2

Lê os CSVs ao iniciar e calcula tudo em memória. A contagem de tokens (truncamento em 512)
usa o tokenizador do BERTimbau e fica em cache em eda_app/.cache/ (1ª vez ~1-2 min); sem o
tokenizador, cai numa estimativa de 4 caracteres/token (marcada como estimativa na tela).
Dependências: flask, pandas, numpy (+ transformers, opcional).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
SPLITS = ["train", "dev", "test"]
DOMINANTES = ["ndmais.com.br"]  # portal de treino do experimento de viés (E3); os demais formam o pool de teste
MAX_LEN = 512

# Evolução dos dados por etapa (registro em ANALISE_EXPLORATORIA.md, seção 11)
ETAPAS = [
    {"etapa": "1. Bruto", "train": 36652, "dev": 9163, "test": 11454, "pos": 879},
    {"etapa": "2. Dedup exato/normalizado", "train": 36650, "dev": 9162, "test": 11453, "pos": 879},
    {"etapa": "3. Filtro sim. TF-IDF ≥ 0,9 (teste)", "train": 36650, "dev": 9162, "test": 11374, "pos": 878},
    {"etapa": "4. Limpeza do rodapé ICL (38 textos)", "train": 36650, "dev": 9162, "test": 11374, "pos": 878},
]
# Termos que podem denunciar a fonte no texto (auditoria de atalho de portal)
TERMOS = ["icl notícias", "jornal conexão", "g1", "leia também", "nsc total", "ndmais", "mpsc", "ministério público"]

app = Flask(__name__, static_folder=str(HERE / "static"))
DF = None            # DataFrame com os 3 splits
TOK = None           # np.array com nº de tokens por linha (ou estimativa)
TOK_ESTIMADO = False
DATA_DIR = None


def carrega(data_dir):
    dfs = []
    for s in SPLITS:
        d = pd.read_csv(Path(data_dir) / f"{s}_bert.csv")
        d["split"] = s
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["processed_text"] = df["processed_text"].fillna("")
    df["chars"] = df["processed_text"].str.len()
    return df


def conta_tokens(df):
    """Nº de tokens (sem truncar) por linha; cache em disco."""
    global TOK_ESTIMADO
    cache = HERE / ".cache" / "tokens.npy"
    chave = HERE / ".cache" / "tokens_len.txt"
    if cache.exists() and chave.exists() and chave.read_text() == str(len(df)):
        return np.load(cache)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        textos = df["processed_text"].tolist()
        out = []
        for i in range(0, len(textos), 1000):
            enc = tok(textos[i:i + 1000], truncation=False, padding=False, add_special_tokens=True)
            out.extend(len(x) for x in enc["input_ids"])
            print(f"  tokens {min(i + 1000, len(textos))}/{len(textos)}", end="\r", flush=True)
        arr = np.array(out)
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, arr)
        chave.write_text(str(len(df)))
        return arr
    except Exception as e:  # sem tokenizador/internet
        print(f"Tokenizador indisponível ({e}); usando estimativa de 4 chars/token.")
        TOK_ESTIMADO = True
        return (df["chars"].to_numpy() / 4 + 2).astype(int)


def filtra(split):
    if split in SPLITS:
        return DF[DF["split"] == split]
    return DF


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/overview")
def overview():
    linhas = []
    for s in SPLITS + ["total"]:
        d = filtra(s)
        pos = int(d["label"].sum())
        linhas.append({"split": s, "n": len(d), "pos": pos, "neg": len(d) - pos,
                       "pct": round(100 * pos / len(d), 3), "portais": int(d["portal"].nunique())})
    dom = DF["portal"].isin(DOMINANTES)
    return jsonify({
        "splits": linhas,
        "etapas": ETAPAS,
        "dominantes": {
            "portais": DOMINANTES,
            "linhas_pct": round(100 * dom.mean(), 2),
            "pos_pct": round(100 * DF.loc[dom, "label"].sum() / DF["label"].sum(), 1),
            "n_fora": int((~dom).sum()), "pos_fora": int(DF.loc[~dom, "label"].sum()),
            "portais_fora": int(DF.loc[~dom, "portal"].nunique()),
        },
    })


@app.route("/api/portals")
def portals():
    d = filtra(request.args.get("split", "all"))
    g = d.groupby("portal")["label"].agg(n="size", pos="sum").reset_index()
    g["pct"] = (100 * g["pos"] / g["n"]).round(2)
    g["neg"] = g["n"] - g["pos"]
    g = g.sort_values("n", ascending=False)
    tot_pos = max(int(g["pos"].sum()), 1)
    g["pos_share"] = (100 * g["pos"] / tot_pos).round(2)
    return jsonify(g.to_dict(orient="records"))


@app.route("/api/lengths")
def lengths():
    d = filtra(request.args.get("split", "all"))
    bins = np.linspace(2, 4.7, 28)  # log10(chars)
    out = {"bins": [round(float(10 ** b)) for b in bins[:-1]], "stats": {}}
    for lbl, nome in [(0, "normal"), (1, "fraude")]:
        x = d.loc[d["label"] == lbl, "chars"].clip(lower=1)
        h, _ = np.histogram(np.log10(x), bins=bins)
        out[nome] = (100 * h / max(len(x), 1)).round(2).tolist()
        out["stats"][nome] = {"n": len(x), "mediana": int(x.median()), "q75": int(x.quantile(0.75)),
                              "max": int(x.max())}
    return jsonify(out)


@app.route("/api/tokens")
def tokens():
    split = request.args.get("split", "all")
    m = (DF["split"] == split).to_numpy() if split in SPLITS else np.ones(len(DF), bool)
    edges = list(range(0, 1537, 64))  # o último bin (1472+) junta a cauda
    out = {"bins": [str(a) for a in edges[:-2]] + ["1472+"], "estimado": TOK_ESTIMADO,
           "max_len": MAX_LEN, "stats": {}}
    for lbl, nome in [(0, "normal"), (1, "fraude")]:
        t = TOK[m & (DF["label"] == lbl).to_numpy()]
        h, _ = np.histogram(np.minimum(t, 1535), bins=edges)
        out[nome] = (100 * h / max(len(t), 1)).round(2).tolist()
        out["stats"][nome] = {"n": len(t), "mediana": int(np.median(t)),
                              "pct_truncado": round(100 * float((t > MAX_LEN).mean()), 1)}
    return jsonify(out)


@app.route("/api/similarity")
def similarity():
    out = {"bins": [], "split": {}, "resumo": []}
    edges = np.round(np.arange(0.70, 1.0001, 0.02), 2)
    out["bins"] = [f"{a:.2f}" for a in edges[:-1]]
    for s in ["dev", "test"]:
        f = DATA_DIR / "saida_checagem" / f"quase_duplicatas_{s}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        h, _ = np.histogram(d["sim"].clip(upper=0.9999), bins=edges)
        out["split"][s] = h.tolist()
        for t in (0.8, 0.9, 0.95, 0.99):
            m = d["sim"] >= t
            out["resumo"].append({"consulta": s, "limiar": t, "pares": int(m.sum()),
                                  "positivos": int(d.loc[m, "label"].sum())})
    return jsonify(out)


@app.route("/api/terms")
def terms():
    t = DF["processed_text"].str.lower()
    rows = []
    for k in TERMOS:
        m = t.str.contains(k, regex=False)
        n = int(m.sum())
        pos = int((m & (DF["label"] == 1)).sum())
        rows.append({"termo": k, "docs": n, "pos": pos, "pct_pos": round(100 * pos / n, 1) if n else 0})
    return jsonify(rows)


@app.route("/api/samples")
def samples():
    d = DF
    if request.args.get("split") in SPLITS:
        d = d[d["split"] == request.args["split"]]
    if request.args.get("portal"):
        d = d[d["portal"] == request.args["portal"]]
    if request.args.get("label") in ("0", "1"):
        d = d[d["label"] == int(request.args["label"])]
    q = request.args.get("q", "").strip().lower()
    if q:
        d = d[d["processed_text"].str.lower().str.contains(q, regex=False)]
    total = len(d)
    page = max(int(request.args.get("page", 0)), 0)
    d = d.iloc[page * 25:(page + 1) * 25]
    return jsonify({"total": total, "page": page, "rows": [
        {"split": r.split, "portal": r.portal, "label": int(r.label), "url": r.url,
         "chars": int(r.chars), "texto": r.processed_text[:320]} for r in d.itertuples()]})


def main():
    global DF, TOK, DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE.parent / "bertimbau_2"), help="pasta com os *_bert.csv")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    DATA_DIR = Path(a.data)
    print("Carregando CSVs...")
    DF = carrega(DATA_DIR)
    print(f"{len(DF)} linhas. Contando tokens (cache se já existir)...")
    TOK = conta_tokens(DF)
    print(f"Pronto: http://127.0.0.1:{a.port}")
    app.run(host="127.0.0.1", port=a.port, debug=False)


if __name__ == "__main__":
    main()
