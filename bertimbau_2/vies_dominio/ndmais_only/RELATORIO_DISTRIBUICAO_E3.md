# Relatório de Preparação de Dados — Experimento E3 (`ndmais_only`)

**Data de Geração:** 2026-10-01 12:56
**Seed Aleatória:** 42
**Portal Dominante (In-Domain):** `ndmais.com.br`

## 1. Conjuntos In-Domain (Treino, Dev e Teste)

| Arquivo | Positivos | Negativos | Total | Razão Real | Observação |
|---|---|---|---|---|---|
| `train_natural.csv` | 270 | 28715 | 28985 | 1:106.4 | Todos os dados ndmais do treino |
| `train_1a64.csv` | 270 | 17280 | 17550 | 1:64.0 | Negativos subamostrados |
| `train_1a10.csv` | 270 | 2700 | 2970 | 1:10.0 | Negativos subamostrados (aninhados ao 1:64) |
| `dev.csv` | 59 | 7292 | 7351 | 1:123.6 | Avaliação e escolha do limiar |
| `test.csv` | 81 | 9010 | 9091 | 1:111.2 | Teste in-domain final |

## 2. Conjuntos de Exposição Interportais (Cross-Portal Pool)

| Arquivo | Teto Pos/Portal | Positivos | Negativos | Total | Razão Real |
|---|---|---|---|---|---|
| `exposicao_1a106.csv` | 12 | 106 | 11273 | 11379 | 1:106.3 |
| `exposicao_1a64.csv` | 29 | 174 | 11136 | 11310 | 1:64.0 |
| `exposicao_1a10.csv` | 29 | 174 | 1740 | 1914 | 1:10.0 |

## 3. Distribuição de Positivos por Portal na Exposição (Teto 29)

| Portal | Positivos Originais no Pool | Positivos Selecionados (Teto 29) |
|---|---|---|
| `iclnoticias.com.br` | 261 | 29 |
| `g1.globo.com` | 62 | 29 |
| `www.nsctotal.com.br` | 47 | 29 |
| `jornalconexao.com.br` | 40 | 29 |
| `www.bbc.com` | 12 | 12 |
| `www.cartacapital.com.br` | 10 | 10 |
| `www.gazetadopovo.com.br` | 9 | 9 |
| `agoralaguna.com.br` | 7 | 7 |
| `www.mpsc.mp.br` | 5 | 5 |
| `www1.folha.uol.com.br` | 3 | 3 |
| `pc.sc.gov.br` | 2 | 2 |
| `www.cnnbrasil.com.br` | 1 | 1 |
| `f5.folha.uol.com.br` | 1 | 1 |
| `lerunica.com.br` | 1 | 1 |
| `intranet.mpsc.mp.br` | 1 | 1 |
| `especiais.gazetadopovo.com.br` | 1 | 1 |
| `www.tcesc.tc.br` | 1 | 1 |
| `agenciabrasil.ebc.com.br` | 1 | 1 |
| `omunicipioblumenau.com.br` | 1 | 1 |
| `portal.mpsc.mp.br` | 1 | 1 |
| `omunicipio.com.br` | 1 | 1 |
