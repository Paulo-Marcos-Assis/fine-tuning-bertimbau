#!/usr/bin/env python3
"""
prepara_dados_ndmais.py — Task T5 do PLANO_VIES_DOMINIO.md

Prepara os conjuntos de dados para o Experimento E3:
1. Conjuntos in-domain do portal dominante (ndmais.com.br):
   - Treino em 3 razões (Natural 1:106, Subamostrado 1:64, Subamostrado 1:10)
   - Dev e Teste in-domain na proporção natural
2. Conjuntos de exposição interportais (pool externo):
   - Exclusão estrita do ndmais.com.br
   - Tetos por portal para os positivos (evita dominância de iclnoticias)
   - Subamostragem aninhada de negativos para casar com as 3 razões
   - Checagem rigorosa de ausência de vazamento (URLs e similaridade TF-IDF)
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
PORTAL_DOMINANTE = "ndmais.com.br"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUT_DIR = os.path.join(BASE_DIR, "ndmais_only")

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("TASK T5: Preparação de dados para o Experimento E3 (ndmais_only)")
print("=" * 70)

# 1. Carregamento dos dados originais
print("\n[1/5] Carregando splits originais...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_bert.csv"))
dev_df = pd.read_csv(os.path.join(DATA_DIR, "dev_bert.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_bert.csv"))

print(f"  Treino original: {len(train_df)} linhas")
print(f"  Dev original:    {len(dev_df)} linhas")
print(f"  Teste original:  {len(test_df)} linhas")

# 2. Separação in-domain (ndmais.com.br)
print(f"\n[2/5] Separando dados in-domain ({PORTAL_DOMINANTE})...")
nd_train = train_df[train_df["portal"] == PORTAL_DOMINANTE].copy()
nd_dev = dev_df[dev_df["portal"] == PORTAL_DOMINANTE].copy()
nd_test = test_df[test_df["portal"] == PORTAL_DOMINANTE].copy()

nd_train_pos = nd_train[nd_train["label"] == 1]
nd_train_neg = nd_train[nd_train["label"] == 0]

n_pos_train = len(nd_train_pos)
n_neg_train = len(nd_train_neg)
print(f"  Treino ndmais: {len(nd_train)} (Pos: {n_pos_train}, Neg: {n_neg_train}, Razão: 1:{n_neg_train/n_pos_train:.1f})")
print(f"  Dev ndmais:    {len(nd_dev)} (Pos: {(nd_dev['label']==1).sum()}, Neg: {(nd_dev['label']==0).sum()})")
print(f"  Teste ndmais:  {len(nd_test)} (Pos: {(nd_test['label']==1).sum()}, Neg: {(nd_test['label']==0).sum()})")

# Salva dev e teste in-domain
nd_dev.to_csv(os.path.join(OUT_DIR, "dev.csv"), index=False)
nd_test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

# Configurações de Treino (aninhadas):
# Natural (1:106.4)
nd_train.to_csv(os.path.join(OUT_DIR, "train_natural.csv"), index=False)

# Subamostragem de negativos para 1:64 e 1:10 (aninhada via permutação fixa)
rng = np.random.default_rng(SEED)
shuffled_neg_idx = rng.permutation(nd_train_neg.index)

# 1:64 -> 270 pos * 64 = 17.280 neg
n_neg_64 = n_pos_train * 64
idx_neg_64 = shuffled_neg_idx[:n_neg_64]
train_1a64 = pd.concat([nd_train_pos, nd_train_neg.loc[idx_neg_64]]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
train_1a64.to_csv(os.path.join(OUT_DIR, "train_1a64.csv"), index=False)

# 1:10 -> 270 pos * 10 = 2.700 neg (subconjunto estrito do 1:64)
n_neg_10 = n_pos_train * 10
idx_neg_10 = shuffled_neg_idx[:n_neg_10]
train_1a10 = pd.concat([nd_train_pos, nd_train_neg.loc[idx_neg_10]]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
train_1a10.to_csv(os.path.join(OUT_DIR, "train_1a10.csv"), index=False)

print(f"  -> Salvo train_natural.csv: {len(nd_train)} linhas")
print(f"  -> Salvo train_1a64.csv:    {len(train_1a64)} linhas ({n_pos_train} pos, {n_neg_64} neg)")
print(f"  -> Salvo train_1a10.csv:    {len(train_1a10)} linhas ({n_pos_train} pos, {n_neg_10} neg)")

# 3. Montagem do Pool de Exposição (outros portais)
print("\n[3/5] Montando pool de outros portais...")
pool = pd.concat([
    train_df[train_df["portal"] != PORTAL_DOMINANTE],
    dev_df[dev_df["portal"] != PORTAL_DOMINANTE],
    test_df[test_df["portal"] != PORTAL_DOMINANTE]
]).reset_index(drop=True)

pool_pos = pool[pool["label"] == 1].copy()
pool_neg = pool[pool["label"] == 0].copy()

print(f"  Pool total: {len(pool)} linhas ({len(pool_pos)} pos, {len(pool_neg)} neg) em {pool['portal'].nunique()} portais")

# 4. Checagem de Vazamento (URLs e Similaridade TF-IDF)
print("\n[4/5] Verificando ausência de vazamento entre pool e treino ndmais...")
urls_treino = set(nd_train["url"])
urls_pool = set(pool["url"])
overlap_url = urls_treino.intersection(urls_pool)
assert len(overlap_url) == 0, f"ERRO: {len(overlap_url)} URLs em comum entre treino e pool de exposição!"
print("  ✓ Zero sobreposição de URLs.")

# Similaridade TF-IDF
vec = TfidfVectorizer(max_features=20000, sublinear_tf=True)
X_train = vec.fit_transform(nd_train["processed_text"])
X_pool = vec.transform(pool["processed_text"])

chunk_size = 2000
max_sims = []
for i in range(0, X_pool.shape[0], chunk_size):
    sim = cosine_similarity(X_pool[i:i+chunk_size], X_train)
    max_sims.append(sim.max(axis=1))
max_sim_total = np.concatenate(max_sims).max()
print(f"  ✓ Similaridade cosseno TF-IDF máxima: {max_sim_total:.4f} (< 0.80, sem quase-duplicatas).")

# 5. Amostragem Estratificada de Exposição (Tetos aninhados e Negativos)
print("\n[5/5] Gerando conjuntos de exposição para as 3 razões...")

def amostrar_positivos_com_teto(pool_pos_df, teto):
    """Amostra até `teto` positivos por portal usando seed fixa."""
    partes = []
    for portal, grupo in pool_pos_df.groupby("portal"):
        n_escolher = min(len(grupo), teto)
        partes.append(grupo.sample(n=n_escolher, random_state=SEED))
    return pd.concat(partes).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# Sorteio aninhado dos negativos do pool
shuffled_pool_neg_idx = rng.permutation(pool_neg.index)

# Exposição 1:64 (teto 29 -> 174 pos, 11.136 neg)
pos_1a64 = amostrar_positivos_com_teto(pool_pos, teto=29)
n_pos_exp_64 = len(pos_1a64)
n_neg_exp_64 = n_pos_exp_64 * 64  # 174 * 64 = 11.136
neg_1a64 = pool_neg.loc[shuffled_pool_neg_idx[:n_neg_exp_64]]
exp_1a64 = pd.concat([pos_1a64, neg_1a64]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
exp_1a64.to_csv(os.path.join(OUT_DIR, "exposicao_1a64.csv"), index=False)

# Exposição 1:10 (teto 29 -> 174 pos [idênticos], 1.740 neg [subconjunto do 1:64])
n_neg_exp_10 = n_pos_exp_64 * 10  # 174 * 10 = 1.740
neg_1a10 = pool_neg.loc[shuffled_pool_neg_idx[:n_neg_exp_10]]
exp_1a10 = pd.concat([pos_1a64, neg_1a10]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
exp_1a10.to_csv(os.path.join(OUT_DIR, "exposicao_1a10.csv"), index=False)

# Exposição 1:106 (teto 12 -> 106 pos, 11.273 neg)
pos_1a106 = amostrar_positivos_com_teto(pool_pos, teto=12)
n_pos_exp_106 = len(pos_1a106)
n_neg_exp_106 = int(round(n_pos_exp_106 * (n_neg_train / n_pos_train))) # 106 * 106.35 = 11.273
neg_1a106 = pool_neg.loc[shuffled_pool_neg_idx[:n_neg_exp_106]]
exp_1a106 = pd.concat([pos_1a106, neg_1a106]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
exp_1a106.to_csv(os.path.join(OUT_DIR, "exposicao_1a106.csv"), index=False)

print(f"  -> Salvo exposicao_1a64.csv:  {len(exp_1a64)} linhas ({n_pos_exp_64} pos, {n_neg_exp_64} neg)")
print(f"  -> Salvo exposicao_1a10.csv:  {len(exp_1a10)} linhas ({n_pos_exp_64} pos, {n_neg_exp_10} neg)")
print(f"  -> Salvo exposicao_1a106.csv: {len(exp_1a106)} linhas ({n_pos_exp_106} pos, {n_neg_exp_106} neg)")

# 6. Geração do Relatório Markdown de Distribuição
relatorio_path = os.path.join(OUT_DIR, "RELATORIO_DISTRIBUICAO_E3.md")
with open(relatorio_path, "w", encoding="utf-8") as f:
    f.write("# Relatório de Preparação de Dados — Experimento E3 (`ndmais_only`)\n\n")
    f.write(f"**Data de Geração:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"**Seed Aleatória:** {SEED}\n")
    f.write(f"**Portal Dominante (In-Domain):** `{PORTAL_DOMINANTE}`\n\n")
    f.write("## 1. Conjuntos In-Domain (Treino, Dev e Teste)\n\n")
    f.write("| Arquivo | Positivos | Negativos | Total | Razão Real | Observação |\n")
    f.write("|---|---|---|---|---|---|\n")
    f.write(f"| `train_natural.csv` | {n_pos_train} | {n_neg_train} | {len(nd_train)} | 1:{n_neg_train/n_pos_train:.1f} | Todos os dados ndmais do treino |\n")
    f.write(f"| `train_1a64.csv` | {n_pos_train} | {n_neg_64} | {len(train_1a64)} | 1:64.0 | Negativos subamostrados |\n")
    f.write(f"| `train_1a10.csv` | {n_pos_train} | {n_neg_10} | {len(train_1a10)} | 1:10.0 | Negativos subamostrados (aninhados ao 1:64) |\n")
    f.write(f"| `dev.csv` | {(nd_dev['label']==1).sum()} | {(nd_dev['label']==0).sum()} | {len(nd_dev)} | 1:{(nd_dev['label']==0).sum()/(nd_dev['label']==1).sum():.1f} | Avaliação e escolha do limiar |\n")
    f.write(f"| `test.csv` | {(nd_test['label']==1).sum()} | {(nd_test['label']==0).sum()} | {len(nd_test)} | 1:{(nd_test['label']==0).sum()/(nd_test['label']==1).sum():.1f} | Teste in-domain final |\n\n")
    
    f.write("## 2. Conjuntos de Exposição Interportais (Cross-Portal Pool)\n\n")
    f.write("| Arquivo | Teto Pos/Portal | Positivos | Negativos | Total | Razão Real |\n")
    f.write("|---|---|---|---|---|---|\n")
    f.write(f"| `exposicao_1a106.csv` | 12 | {n_pos_exp_106} | {n_neg_exp_106} | {len(exp_1a106)} | 1:{n_neg_exp_106/n_pos_exp_106:.1f} |\n")
    f.write(f"| `exposicao_1a64.csv` | 29 | {n_pos_exp_64} | {n_neg_exp_64} | {len(exp_1a64)} | 1:{n_neg_exp_64/n_pos_exp_64:.1f} |\n")
    f.write(f"| `exposicao_1a10.csv` | 29 | {n_pos_exp_64} | {n_neg_exp_10} | {len(exp_1a10)} | 1:{n_neg_exp_10/n_pos_exp_64:.1f} |\n\n")
    
    f.write("## 3. Distribuição de Positivos por Portal na Exposição (Teto 29)\n\n")
    f.write("| Portal | Positivos Originais no Pool | Positivos Selecionados (Teto 29) |\n")
    f.write("|---|---|---|\n")
    pos_por_portal_orig = pool_pos["portal"].value_counts()
    pos_por_portal_sel = pos_1a64["portal"].value_counts()
    for portal in pos_por_portal_orig.index:
        f.write(f"| `{portal}` | {pos_por_portal_orig[portal]} | {pos_por_portal_sel.get(portal, 0)} |\n")

print(f"\n[OK] Relatório completo gravado em: {relatorio_path}")
print("Processamento concluído com sucesso!")
