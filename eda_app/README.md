# EDA interativa (localhost)

Gráficos e tabelas interativos sobre os splits atuais de `bertimbau_2/` (visão geral, portais × classe, tamanho e truncamento em 512 tokens, quase-duplicatas, termos de assinatura de portal e explorador de textos). O filtro de split no topo vale para portais, tamanho e textos.

```
pip install flask pandas numpy          # transformers é opcional (contagem exata de tokens)
python eda_app/app.py                   # http://127.0.0.1:8050
python eda_app/app.py --port 8060 --data bertimbau_2
```

- Lê `train/dev/test_bert.csv` ao iniciar; nada é gravado fora de `eda_app/.cache/` (cache dos tokens, 1ª execução ~1–2 min).
- A aba de quase-duplicatas usa `bertimbau_2/saida_checagem/quase_duplicatas_{dev,test}.csv` (gerados com o teste antes do filtro de 79 linhas).
- `static/chart.umd.min.js` é o Chart.js 4.4.3 (incluído para funcionar sem internet).
