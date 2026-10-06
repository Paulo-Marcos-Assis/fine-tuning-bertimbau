# Tarefa futura — comparar outros modelos com o BERTimbau no fine-tuning

**Status:** ideia registrada em 2026-09-30, **não iniciada**. Origem: lista do usuário em `embeddings_to_compare.txt` (fora do repositório). Só começar depois do notebook principal e do teste de viés (`PLANO_VIES_DOMINIO.md`), ou quando o usuário pedir.

## 1. Candidatos

| Modelo | Tipo | Fonte | Observação |
|---|---|---|---|
| **NorBERTo** | ModernBERT treinado para português (corpus de 331 bi de tokens) | Silva et al. 2026, PROPOR 2026, vol. 1, p. 183–193 | id no Hugging Face: a confirmar |
| **ModBERTBr** | ModernBERT para português brasileiro | Wu & Garcia, ENIAC 2025, p. 2044–2055, DOI 10.5753/eniac.2025.14516 | id no Hugging Face: a confirmar |
| **BERTugues** | BERT pré-treinado para português brasileiro | Mazza Zago & Agnoletti dos Santos Pedotti, 2024, Semina: Ciências Exatas e Tecnológicas 45, e50630, DOI 10.5433/1679-0375.2024.v45.50630 | id no Hugging Face: a confirmar |
| **multilingual-e5-large** | modelo de embeddings multilíngue | https://huggingface.co/intfloat/multilingual-e5-large | usado via `sentence-transformers` ou `transformers`; conferir na model card o formato de entrada (prefixos) e o tamanho máximo |
| `sentence-transformers` | biblioteca (não é um modelo) | https://sbert.net/ (`pip install -U sentence-transformers`) | ferramenta para carregar/usar o e5 |

Referência atual: `neuralmind/bert-base-portuguese-cased` (BERTimbau base).

## 2. Por que vale testar e justificativa empírica

1. **Evolução arquitetural:** Modelos mais recentes treinados especificamente para português (como os baseados em ModernBERT) incorporam avanços arquiteturais (Rotary Embeddings, FlashAttention, vocabulários expandidos) que podem superar o BERTimbau base na mesma tarefa.
2. **Justificativa empírica de Contexto Longo (Medições da PARTE 3.1 no treino):**
   A execução da PARTE 3.1 no notebook `bertimbau_tuning.ipynb` comprovou uma assimetria severa de comprimento entre as classes:
   - **Notícias normais ($n=36.088$):** mediana = 382 tokens, q75 = 605 tokens, **33,2%** sofrem truncamento em 512 tokens.
   - **Notícias de fraude ($n=562$):** mediana = **738 tokens**, q75 = **1.086 tokens**, **70,6%** sofrem truncamento em 512 tokens.

   **Impacto prático e metodológico:**
   - Mais de 7 em cada 10 notícias de fraude têm o texto cortado em 512 tokens no BERTimbau, gerando um "ponto cego" para trechos frequentemente situados no final de reportagens investigativas (despachos judiciais, conclusões de inquérito policial e pareceres de órgãos de controle).
   - Como o 3º quartil (q75) das fraudes é de 1.086 tokens, arquiteturas que aceitam janelas de contexto estendidas (ex.: 1.024, 2.048 ou 4.096 tokens) cobrirão a quase totalidade dos textos de fraude sem nenhuma perda de sinal informativo.

## 3. Tasks

| # | Task | Aceite |
|---|---|---|
| F1 | Para cada candidato: confirmar o id no Hugging Face, licença, tamanho, contexto máximo e tokenizador | tabela preenchida na seção 1 |
| F2 | Parametrizar o notebook principal pelo nome do modelo (`AutoTokenizer`/`AutoModelForSequenceClassification`, `max_length`, prefixos se o modelo exigir) sem mudar o restante | o BERTimbau reproduz o resultado atual |
| F3 | Treinar cada candidato no **mesmo protocolo** (mesmos splits, seed 42, loss ponderada, seleção pelo dev, avaliação única no teste) e salvar predições no formato da T1 do plano de viés | tabela modelo × {AP, F1 no limiar do dev, recall, IC} |
| F4 | Nos modelos com contexto longo, repetir com `max_length` maior (ex.: 1024 ou 2048) e comparar com 512 | efeito do truncamento medido |
| F5 | Repetir o teste de viés de domínio (E3) só para o melhor candidato, se o usuário quiser | resultado registrado no plano de viés |

**Ressalva:** antes de implementar F2–F5, confirmar com o usuário a lista final de modelos e o orçamento de GPU.
