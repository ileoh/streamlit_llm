# Guia: Deploy do Modelo de Análise de Sentimento

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
pip install -r requirements.txt
```

### 3. Baixar Recursos NLTK
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## 🔧 Estrutura do Projeto

```
projeto/
├── sentiment_pipeline.pkl    # Modelo treinado
├── app_simples.py           # Aplicativo Streamlit principal
├── requirements.txt          # Dependências (otimizado para deploy)
├── .streamlit/
│   └── config.toml         # Configuração do Streamlit
└── passos_deploy.md        # Este guia
```

## 🎯 Passos para Deploy Local

### 1. Verificar Arquivos
- Confirme que `sentiment_pipeline.pkl` está na raiz do projeto
- Verifique se `app_simples.py` está configurado corretamente

### 2. Executar Localmente
```bash
streamlit run app_simples.py
```

## ☁️ Deploy no Streamlit Cloud

### 1. Preparação
- Certifique-se de que o código está no GitHub
- Verifique se `requirements.txt` está otimizado (sem `sentencepiece` explícito)
- Confirme que `.streamlit/config.toml` está presente

### 2. Deploy no Streamlit Cloud
1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Faça login com sua conta GitHub
3. Clique em "New app"
4. Selecione o repositório e arquivo `app_simples.py`
5. Clique em "Deploy!"

### 3. Configurações Importantes
- **Arquivo principal**: `app_simples.py`
- **Requirements**: `requirements.txt` (otimiz
- **Configuração**: `.streamlit/config.toml`

## 🛠️ Solução de Problemas

### Erro "Error installing requirements"
**Causa**: Dependências que precisam de compilação (como `sentencepiece`)
**Solução**: Remover `sentencepiece` do `requirements.txt` - será gerenciado automaticamente

### Erro de NLTK
**Causa**: Recursos NLTK não baixados
**Solução**: O app inclui download automático via `@st.cache_resource`

### Problemas de Deploy
1. Verifique se todos os arquivos estão no GitHub
2. Confirme que `requirements.txt` está limpo
3. Aguarde alguns minutos para o deploy inicial

## 📝 Checklist Final

### Para Deploy Local
- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Recursos NLTK baixados
- [ ] Modelo pickle na raiz
- [ ] App funcionando localmente (`streamlit run app_simples.py`)

### Para Deploy Cloud
- [ ] Código no GitHub
- [ ] `requirements.txt` otimizado (sem `sentencepiece`)
- [ ] `.streamlit/config.toml` presente
- [ ] Deploy iniciado no Streamlit Cloud
- [ ] App funcionando online

## 🎉 Resultado

### Local
O aplicativo estará disponível em: `http://localhost:8501`

### Cloud
O aplicativo estará disponível em: `https://[seu-app].streamlit.app`

**Funcionalidades:**
- ✅ Análise individual de texto
- ✅ Interface moderna e responsiva
- ✅ Tratamento de erros robusto
- ✅ Download automático de recursos NLTK
- ✅ Cache de recursos para performance

## 📊 Monitoramento

### Logs do Streamlit Cloud
- Acesse "Manage App" no Streamlit Cloud
- Verifique logs em caso de erro
- Monitore performance e uso

### Atualizações
- Push para GitHub atualiza automaticamente o deploy
- Aguarde 2-5 minutos para propagação
