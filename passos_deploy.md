# Guia: Do Modelo Pickle para Streamlit Local

## 📋 Pré-requisitos

### 1. Ambiente Virtual
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (macOS/Linux)
source .venv/bin/activate

# Ativar ambiente (Windows)
.venv\Scripts\activate
```

### 2. Instalar Dependências
```bash
# Instalar bibliotecas necessárias
pip install streamlit pandas scikit-learn nltk pysentimiento transformers torch

# Ou usar o requirements.txt
pip install -r requirements.txt
```

### 3. Baixar Recursos NLTK
```bash
python -c "import nltk; nltk.download('stopwords')"
```

## 🔧 Estrutura do Projeto

```
projeto/
├── sentiment_pipeline.pkl    # Modelo treinado
├── app.py                    # Aplicativo Streamlit
├── requirements.txt          # Dependências
└── README.md               # Documentação
```

## 🎯 Passos para Criar o App

### 1. Verificar se o Modelo Existe
- Confirme que `sentiment_pipeline.pkl` está na raiz do projeto

### 2. Criar o App Streamlit (`app.py`)
- Incluir todas as funções e classes do notebook
- Usar `@st.cache_resource` para carregar modelo e analyzer
- Implementar interface para análise individual e em lote

### 3. Executar Localmente
```bash
streamlit run app.py
```

## 📝 Checklist Final

- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas
- [ ] Recursos NLTK baixados
- [ ] Modelo pickle na raiz
- [ ] Funções e classes do notebook incluídas no app.py
- [ ] App Streamlit criado com todas as dependências
- [ ] Teste local funcionando

## 🎉 Resultado

O aplicativo estará disponível em: `http://localhost:8501`

**Funcionalidades:**
- ✅ Análise individual de texto
- ✅ Análise em lote com upload CSV
- ✅ Download dos resultados
- ✅ Interface moderna e responsiva
