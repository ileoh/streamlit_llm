import streamlit as st
import pickle
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from pysentimiento import create_analyzer
import pandas as pd

# Funções necessárias
def limpar_texto_completo(texto):
    if isinstance(texto, str):
        texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
        texto = re.sub(r'@\w+', '', texto)
        texto = re.sub(r'#\w+', '', texto)
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = re.sub(r'\d+', '', texto)
        texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
        texto = ' '.join(texto.split())
        return texto.lower()
    return ''

def remover_stop_words(texto):
    if isinstance(texto, str):
        stop_words = set(stopwords.words('portuguese'))
        palavras = texto.split()
        texto_sem_stopwords = [palavra for palavra in palavras if palavra not in stop_words]
        return ' '.join(texto_sem_stopwords)
    return ''

def classificar_sentimento_binario(texto, threshold=0.55):
    resultado = analyzer.predict(texto)
    probas = resultado.probas
    if resultado.output == "POS":
        return pd.Series(["positivo", round(probas["POS"], 3)])
    elif resultado.output == "NEG":
        return pd.Series(["negativo", round(probas["NEG"], 3)])
    else:
        if probas["POS"] >= threshold:
            return pd.Series(["positivo", round(probas["POS"], 3)])
        elif probas["NEG"] >= threshold:
            return pd.Series(["negativo", round(probas["NEG"], 3)])
        else:
            return pd.Series(["incerto", max(round(probas["POS"], 3), round(probas["NEG"], 3))])

class SentimentPipeline:
    def __init__(self):
        pass
    def processar_texto(self, texto):
        texto_limpo = limpar_texto_completo(texto)
        texto_sem_stop = remover_stop_words(texto_limpo)
        sentimento, prob = classificar_sentimento_binario(texto_sem_stop)
        return {"sentimento": sentimento, "probabilidade": prob}

# Carregar componentes
@st.cache_resource
def carregar_analyzer():
    return create_analyzer(task="sentiment", lang="pt")

@st.cache_resource
def carregar_modelo():
    with open('sentiment_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

# Interface
st.title("Análise de Sentimento")
texto = st.text_area("Digite o texto:")
if st.button("Prever"):
    if texto.strip():
        analyzer = carregar_analyzer()
        modelo = carregar_modelo()
        resultado = modelo.processar_texto(texto)
        st.write(f"Sentimento: {resultado['sentimento']}")
        st.write(f"Confiança: {resultado['probabilidade']:.1%}")
    else:
        st.warning("Digite um texto") 