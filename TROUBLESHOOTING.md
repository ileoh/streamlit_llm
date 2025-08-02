# 🔧 Troubleshooting - Solução de Problemas

## 📋 Problemas Comuns e Soluções

### 1. Erro de Importação do NLTK
**Problema:** `ModuleNotFoundError: No module named 'nltk'`
**Solução:**
```bash
pip install nltk
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 2. Erro de Dados do NLTK
**Problema:** `LookupError: Resource stopwords not found`
**Solução:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

### 3. Erro do PySentimiento
**Problema:** `ModuleNotFoundError: No module named 'pysentimiento'`
**Solução:**
```bash
pip install pysentimiento
```

### 4. Erro do Transformers
**Problema:** `ModuleNotFoundError: No module named 'transformers'`
**Solução:**
```bash
pip install transformers torch
```

### 5. Erro do WordCloud
**Problema:** `ModuleNotFoundError: No module named 'wordcloud'`
**Solução:**
```bash
pip install wordcloud
```

### 6. Erro de Compatibilidade do Torch
**Problema:** Versões incompatíveis do PyTorch
**Solução:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 7. Erro de Memória
**Problema:** `CUDA out of memory` ou `RuntimeError: CUDA out of memory`
**Solução:**
- Use CPU em vez de GPU
- Reduza o batch size
- Feche outros aplicativos

### 8. Erro do Streamlit
**Problema:** `ModuleNotFoundError: No module named 'streamlit'`
**Solução:**
```bash
pip install streamlit
```

### 9. Erro de Versão do Python
**Problema:** Versão do Python incompatível
**Solução:**
- Use Python 3.8 ou superior
- Verifique com: `python --version`

### 10. Erro de Dependências Circulares
**Problema:** Conflitos entre versões
**Solução:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 🚀 Setup Automático

Execute o script de setup:
```bash
python setup.py
```

## 📦 Instalação Manual

1. **Instalar dependências básicas:**
```bash
pip install streamlit pandas numpy scikit-learn
```

2. **Instalar NLP:**
```bash
pip install nltk pysentimiento transformers torch
```

3. **Instalar visualização:**
```bash
pip install plotly seaborn matplotlib wordcloud
```

4. **Baixar dados do NLTK:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## 🔍 Verificação de Instalação

Execute este código para verificar se tudo está funcionando:

```python
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from pysentimiento import create_analyzer
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sklearn

print("✅ Todas as importações funcionaram!")
```

## 🌐 Problemas de Rede

Se houver problemas de download:

1. **Use um proxy ou VPN**
2. **Configure pip para usar mirrors locais**
3. **Baixe os modelos offline**

## 💾 Limpeza de Cache

Se houver problemas de cache:

```bash
# Limpar cache do pip
pip cache purge

# Limpar cache do streamlit
streamlit cache clear

# Remover e reinstalar
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

## 📞 Suporte

Se os problemas persistirem:

1. Verifique a versão do Python: `python --version`
2. Verifique a versão do pip: `pip --version`
3. Execute: `pip list` para ver as versões instaladas
4. Consulte a documentação oficial das bibliotecas 