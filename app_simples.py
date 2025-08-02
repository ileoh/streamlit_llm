import streamlit as st
import pickle
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from pysentimiento import create_analyzer
import pandas as pd

# Setup NLTK
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

setup_nltk()

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
        try:
            stop_words = set(stopwords.words('portuguese'))
            palavras = texto.split()
            texto_sem_stopwords = [palavra for palavra in palavras if palavra not in stop_words]
            return ' '.join(texto_sem_stopwords)
        except:
            return texto
    return ''

def classificar_sentimento_binario(texto, threshold=0.55):
    try:
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
    except Exception as e:
        st.error(f"Erro na análise: {str(e)}")
        return pd.Series(["erro", 0.0])

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
    try:
        return create_analyzer(task="sentiment", lang="pt")
    except Exception as e:
        st.error(f"Erro ao carregar analyzer: {str(e)}")
        return None

@st.cache_resource
def carregar_modelo():
    try:
        with open('sentiment_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None

# Interface
st.title("Análise de Sentimento")
st.markdown("---")

# Carregar componentes
analyzer = carregar_analyzer()
modelo = carregar_modelo()

if analyzer is None or modelo is None:
    st.error("Erro ao carregar componentes necessários.")
else:
    texto = st.text_area("Digite o texto para análise:", height=150)
    
    if st.button("Analisar", type="primary"):
        if texto.strip():
            with st.spinner("Analisando sentimento..."):
                resultado = modelo.processar_texto(texto)
                
                st.markdown("### Resultado da Análise")
                
                if resultado['sentimento'] == 'positivo':
                    st.success(f"✅ **Sentimento:** {resultado['sentimento'].title()}")
                elif resultado['sentimento'] == 'negativo':
                    st.error(f"❌ **Sentimento:** {resultado['sentimento'].title()}")
                else:
                    st.warning(f"⚠️ **Sentimento:** {resultado['sentimento'].title()}")
                
                st.info(f"📊 **Confiança:** {resultado['probabilidade']:.1%}")
                st.progress(resultado['probabilidade'])
        else:
            st.warning("⚠️ Digite um texto para análise")

st.markdown("---")
st.markdown("*Desenvolvido com Streamlit e IA Generativa*") 