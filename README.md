# 🤖 Análise de Sentimento - App Streamlit

Aplicativo simples para análise de sentimento usando o modelo treinado com PySentimiento.

## 🚀 Como Executar

1. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

2. **Baixar recursos do NLTK (necessário na primeira execução):**
```bash
python -c "import nltk; nltk.download('stopwords')"
```

3. **Testar se tudo está funcionando (opcional):**
```bash
python test_app.py
```

4. **Executar o aplicativo:**
```bash
streamlit run app.py
```

## 📋 Funcionalidades

- **Análise Individual**: Digite um texto e veja o sentimento
- **Análise em Lote**: Upload de arquivo CSV para análise múltipla
- **Processamento de Texto**: Limpeza automática e remoção de stop words
- **Download**: Baixe os resultados em CSV

## 📁 Arquivos Necessários

- `app.py` - Aplicativo principal
- `sentiment_pipeline.pkl` - Modelo treinado (opcional)
- `requirements.txt` - Dependências
- `test_app.py` - Script de teste

## 💡 Como Usar

1. Abra o aplicativo no navegador
2. Digite um texto ou faça upload de um CSV
3. Clique em "Analisar Sentimento"
4. Veja os resultados e faça download se necessário

## 🔧 Tecnologias Utilizadas

- **Streamlit**: Interface web
- **PySentimiento**: Modelo de análise de sentimento
- **NLTK**: Processamento de linguagem natural
- **Pandas**: Manipulação de dados

## ⚠️ Notas Importantes

- O aplicativo funciona mesmo sem o arquivo `sentiment_pipeline.pkl`
- Na primeira execução, o modelo será baixado automaticamente
- Para análise em lote, use colunas com nomes contendo 'text' ou 'comentario'
- O script `test_app.py` pode ser usado para verificar se tudo está funcionando