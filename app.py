import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from pysentimiento import create_analyzer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from datetime import datetime
import time

# Configuração da página
st.set_page_config(
    page_title="🤖 Análise de Sentimento Avançada",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #ffc107; }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Configurações")
    
    # Modo de análise
    modo_analise = st.selectbox(
        "🎯 Modo de Análise",
        ["Individual", "Lote", "Dashboard Completo", "Comparação", "Tendências"]
    )
    
    # Configurações avançadas
    with st.expander("🔧 Configurações Avançadas"):
        threshold = st.slider("📊 Threshold de Confiança", 0.0, 1.0, 0.55, 0.05)
        max_textos = st.number_input("📝 Máximo de Textos", 1, 1000, 100)
        mostrar_detalhes = st.checkbox("🔍 Mostrar Detalhes do Processamento", True)
        auto_analise = st.checkbox("⚡ Análise Automática", False)
    
    # Informações do modelo
    with st.expander("📊 Informações do Modelo"):
        st.write("**Modelo:** PySentimiento")
        st.write("**Idioma:** Português")
        st.write("**Tarefa:** Análise de Sentimento")
        st.write("**Classes:** Positivo, Negativo, Neutro")

# Funções necessárias do notebook
def limpar_texto_completo(texto):
    """
    Limpa o texto, removendo URLs, menções, hashtags, caracteres especiais,
    pontuações, números, espaços extras e acentos, convertendo para lowercase.
    """
    if isinstance(texto, str):
        # Remove URLs
        texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
        # Remove menções (@)
        texto = re.sub(r'@\w+', '', texto)
        # Remove hashtags (#)
        texto = re.sub(r'#\w+', '', texto)
        # Remove caracteres especiais e pontuações, mantendo espaços
        texto = re.sub(r'[^\w\s]', '', texto)
        # Remove números
        texto = re.sub(r'\d+', '', texto)
        # Remove acentos
        texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
        # Remove espaços extras
        texto = ' '.join(texto.split())
        return texto.lower()
    return ''

def remover_stop_words(texto):
    """
    Remove stop words de um texto.
    """
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
        # Se for neutro, compara POS e NEG e decide pela maior, se confiável
        if probas["POS"] >= threshold:
            return pd.Series(["positivo", round(probas["POS"], 3)])
        elif probas["NEG"] >= threshold:
            return pd.Series(["negativo", round(probas["NEG"], 3)])
        else:
            return pd.Series(["incerto", max(round(probas["POS"], 3), round(probas["NEG"], 3))])

# Classe SentimentPipeline do notebook
class SentimentPipeline:
    def __init__(self):
        pass

    def processar_texto(self, texto):
        texto_limpo = limpar_texto_completo(texto)
        texto_sem_stop = remover_stop_words(texto_limpo)
        sentimento, prob = classificar_sentimento_binario(texto_sem_stop)
        return {
            "texto_original": texto,
            "texto_processado": texto_sem_stop,
            "sentimento": sentimento,
            "probabilidade": prob,
            "timestamp": datetime.now()
        }

    def processar_dataframe(self, df, coluna="comentario_limpo"):
        resultados = df[coluna].apply(self.processar_texto)
        return pd.DataFrame(resultados.tolist())

# Carregar componentes
@st.cache_resource
def carregar_analyzer():
    try:
        return create_analyzer(task="sentiment", lang="pt")
    except Exception as e:
        st.error(f"Erro ao carregar o analyzer: {e}")
        return None

@st.cache_resource
def carregar_modelo():
    try:
        with open('sentiment_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Inicializar componentes
analyzer = carregar_analyzer()
modelo = carregar_modelo()

if analyzer is None or modelo is None:
    st.error("❌ Não foi possível carregar os componentes necessários.")
    st.stop()

# Título principal
st.markdown('<h1 class="main-header">🤖 Análise de Sentimento Avançada</h1>', unsafe_allow_html=True)

# Funções de visualização
def criar_grafico_sentimentos(df):
    """Cria gráfico de pizza com distribuição de sentimentos"""
    fig = px.pie(
        df, 
        names='sentimento', 
        title='Distribuição de Sentimentos',
        color_discrete_map={'positivo': '#28a745', 'negativo': '#dc3545', 'incerto': '#ffc107'}
    )
    fig.update_layout(height=400)
    return fig

def criar_grafico_confianca(df):
    """Cria histograma de confiança"""
    fig = px.histogram(
        df, 
        x='probabilidade', 
        color='sentimento',
        title='Distribuição de Confiança por Sentimento',
        nbins=20
    )
    fig.update_layout(height=400)
    return fig

def criar_wordcloud(textos, titulo):
    """Cria wordcloud dos textos"""
    if not textos:
        return None
    
    texto_completo = ' '.join(textos)
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis'
    ).generate(texto_completo)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(titulo)
    
    return fig

def criar_metricas_avancadas(df):
    """Cria métricas avançadas"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total de Análises", len(df))
    
    with col2:
        st.metric("📈 Confiança Média", f"{df['probabilidade'].mean():.1%}")
    
    with col3:
        st.metric("🎯 Maior Confiança", f"{df['probabilidade'].max():.1%}")
    
    with col4:
        st.metric("📉 Menor Confiança", f"{df['probabilidade'].min():.1%}")

# Interface principal baseada no modo selecionado
if modo_analise == "Individual":
    st.header("📝 Análise Individual")
    
    # Input de texto
    col1, col2 = st.columns([3, 1])
    
    with col1:
        texto = st.text_area(
            "Digite o texto para análise:",
            placeholder="Ex: Este produto é muito bom, recomendo!",
            height=120
        )
    
    with col2:
        st.write("**Exemplos:**")
        exemplos = [
            "Produto excelente, superou minhas expectativas!",
            "Não gostei nada, péssima qualidade.",
            "Mais ou menos, poderia ser melhor."
        ]
        for i, exemplo in enumerate(exemplos, 1):
            if st.button(f"Exemplo {i}", key=f"ex_{i}"):
                st.session_state.texto_exemplo = exemplo
    
    # Análise automática
    if auto_analise and texto.strip():
        with st.spinner("Analisando automaticamente..."):
            time.sleep(0.5)
            resultado = modelo.processar_texto(texto)
            st.session_state.resultado = resultado
    
    # Botão de análise
    if st.button("🔍 Analisar Sentimento", type="primary", use_container_width=True):
        if texto.strip():
            with st.spinner("Analisando sentimento..."):
                try:
                    # Processar o texto
                    resultado = modelo.processar_texto(texto)
                    st.session_state.resultado = resultado
                    
                    # Exibir resultados
                    st.success("✅ Análise concluída!")
                    
                    # Métricas principais
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Sentimento</h3>
                            <p class="{resultado['sentimento']}">{resultado['sentimento'].upper()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Confiança</h3>
                            <p>{resultado['probabilidade']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        if resultado["sentimento"] == "positivo":
                            st.success("✅ Positivo")
                        elif resultado["sentimento"] == "negativo":
                            st.error("❌ Negativo")
                        else:
                            st.warning("⚠️ Incerto")
                    
                    # Detalhes do processamento
                    if mostrar_detalhes:
                        with st.expander("🔍 Detalhes do Processamento", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Texto Original:**")
                                st.code(resultado["texto_original"])
                                
                                st.write("**Texto Processado:**")
                                st.code(resultado["texto_processado"])
                            
                            with col2:
                                st.write("**Informações da Análise:**")
                                st.json({
                                    "sentimento": resultado["sentimento"],
                                    "probabilidade": resultado["probabilidade"],
                                    "timestamp": str(resultado["timestamp"])
                                })
                    
                except Exception as e:
                    st.error(f"Erro na análise: {e}")
        else:
            st.warning("⚠️ Por favor, digite um texto para análise.")

elif modo_analise == "Lote":
    st.header("📊 Análise em Lote")
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Upload de arquivo CSV:",
        type=['csv'],
        help="O arquivo deve ter uma coluna com os textos para análise"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Arquivo carregado:**")
            st.dataframe(df.head())
            
            # Selecionar coluna
            coluna_texto = st.selectbox(
                "Selecione a coluna com os textos:",
                df.columns.tolist()
            )
            
            # Configurações de processamento
            col1, col2 = st.columns(2)
            with col1:
                processar_todos = st.checkbox("Processar todos os textos", True)
            with col2:
                if not processar_todos:
                    num_textos = st.number_input("Número de textos", 1, len(df), min(100, len(df)))
            
            if st.button("📈 Iniciar Análise em Lote", type="primary"):
                if processar_todos:
                    df_processar = df
                else:
                    df_processar = df.head(num_textos)
                
                # Barra de progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Processando arquivo..."):
                    try:
                        # Processar textos
                        resultados = []
                        total = len(df_processar)
                        
                        for idx, row in df_processar.iterrows():
                            resultado = modelo.processar_texto(row[coluna_texto])
                            resultados.append(resultado)
                            
                            # Atualizar progresso
                            progress = (idx + 1) / total
                            progress_bar.progress(progress)
                            status_text.text(f"Processando... {idx + 1}/{total}")
                        
                        # Criar DataFrame de resultados
                        df_resultados = pd.DataFrame(resultados)
                        df_final = pd.concat([df_processar, df_resultados], axis=1)
                        
                        st.success(f"✅ Análise concluída! {len(df_resultados)} textos processados.")
                        
                        # Métricas avançadas
                        criar_metricas_avancadas(df_resultados)
                        
                        # Visualizações
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.plotly_chart(criar_grafico_sentimentos(df_resultados), use_container_width=True)
                        
                        with col2:
                            st.plotly_chart(criar_grafico_confianca(df_resultados), use_container_width=True)
                        
                        # Wordclouds
                        st.subheader("☁️ Nuvens de Palavras")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            textos_positivos = df_resultados[df_resultados['sentimento'] == 'positivo']['texto_processado'].tolist()
                            if textos_positivos:
                                fig = criar_wordcloud(textos_positivos, "Palavras Positivas")
                                if fig:
                                    st.pyplot(fig)
                        
                        with col2:
                            textos_negativos = df_resultados[df_resultados['sentimento'] == 'negativo']['texto_processado'].tolist()
                            if textos_negativos:
                                fig = criar_wordcloud(textos_negativos, "Palavras Negativas")
                                if fig:
                                    st.pyplot(fig)
                        
                        # Tabela de resultados
                        st.subheader("📋 Resultados Detalhados")
                        st.dataframe(df_final)
                        
                        # Download dos resultados
                        csv = df_final.to_csv(index=False)
                        st.download_button(
                            label="📥 Download dos Resultados (CSV)",
                            data=csv,
                            file_name=f"analise_sentimento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Erro no processamento: {e}")
                        
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")

elif modo_analise == "Dashboard Completo":
    st.header("📊 Dashboard Completo")
    
    # Carregar dados de exemplo
    try:
        df_exemplo = pd.read_csv('comments_amazon.csv')
        st.write("**Dados de Exemplo Carregados:**")
        st.dataframe(df_exemplo.head())
        
        if st.button("🚀 Executar Análise Completa", type="primary"):
            with st.spinner("Executando análise completa..."):
                # Processar dados
                df_processar = df_exemplo.head(500)  # Limitar para performance
                resultados = []
                
                progress_bar = st.progress(0)
                for idx, row in df_processar.iterrows():
                    resultado = modelo.processar_texto(row['Comentario'])
                    resultados.append(resultado)
                    progress_bar.progress((idx + 1) / len(df_processar))
                
                df_resultados = pd.DataFrame(resultados)
                
                # Dashboard completo
                st.success("✅ Dashboard gerado com sucesso!")
                
                # Métricas principais
                criar_metricas_avancadas(df_resultados)
                
                # Gráficos
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(criar_grafico_sentimentos(df_resultados), use_container_width=True)
                with col2:
                    st.plotly_chart(criar_grafico_confianca(df_resultados), use_container_width=True)
                
                # Análise temporal (se houver timestamp)
                if 'timestamp' in df_resultados.columns:
                    df_resultados['timestamp'] = pd.to_datetime(df_resultados['timestamp'])
                    df_resultados['hora'] = df_resultados['timestamp'].dt.hour
                    
                    fig = px.line(
                        df_resultados.groupby('hora')['sentimento'].value_counts().unstack(fill_value=0),
                        title='Tendência de Sentimentos por Hora'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Wordclouds
                st.subheader("☁️ Análise de Palavras")
                col1, col2 = st.columns(2)
                
                with col1:
                    textos_positivos = df_resultados[df_resultados['sentimento'] == 'positivo']['texto_processado'].tolist()
                    if textos_positivos:
                        fig = criar_wordcloud(textos_positivos, "Palavras Positivas")
                        if fig:
                            st.pyplot(fig)
                
                with col2:
                    textos_negativos = df_resultados[df_resultados['sentimento'] == 'negativo']['texto_processado'].tolist()
                    if textos_negativos:
                        fig = criar_wordcloud(textos_negativos, "Palavras Negativas")
                        if fig:
                            st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Erro ao carregar dados de exemplo: {e}")

elif modo_analise == "Comparação":
    st.header("🔄 Comparação de Textos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        texto1 = st.text_area("Texto 1:", height=150)
    
    with col2:
        texto2 = st.text_area("Texto 2:", height=150)
    
    if st.button("🔄 Comparar Sentimentos", type="primary"):
        if texto1.strip() and texto2.strip():
            with st.spinner("Comparando sentimentos..."):
                resultado1 = modelo.processar_texto(texto1)
                resultado2 = modelo.processar_texto(texto2)
                
                # Comparação
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Texto 1")
                    st.write(f"**Sentimento:** {resultado1['sentimento'].upper()}")
                    st.write(f"**Confiança:** {resultado1['probabilidade']:.1%}")
                    st.write(f"**Texto:** {texto1[:100]}...")
                
                with col2:
                    st.subheader("📝 Texto 2")
                    st.write(f"**Sentimento:** {resultado2['sentimento'].upper()}")
                    st.write(f"**Confiança:** {resultado2['probabilidade']:.1%}")
                    st.write(f"**Texto:** {texto2[:100]}...")
                
                # Gráfico de comparação
                df_comparacao = pd.DataFrame([
                    {'Texto': 'Texto 1', 'Sentimento': resultado1['sentimento'], 'Confiança': resultado1['probabilidade']},
                    {'Texto': 'Texto 2', 'Sentimento': resultado2['sentimento'], 'Confiança': resultado2['probabilidade']}
                ])
                
                fig = px.bar(
                    df_comparacao, 
                    x='Texto', 
                    y='Confiança', 
                    color='Sentimento',
                    title='Comparação de Confiança'
                )
                st.plotly_chart(fig, use_container_width=True)

elif modo_analise == "Tendências":
    st.header("📈 Análise de Tendências")
    
    st.info("🔍 Esta funcionalidade permite analisar tendências temporais nos sentimentos.")
    
    # Simulação de dados temporais
    if st.button("📊 Gerar Análise de Tendências", type="primary"):
        # Simular dados temporais
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Simular sentimentos com tendência
        sentimentos = []
        for i in range(len(dates)):
            if i < len(dates) // 2:
                # Primeira metade: mais positivos
                sentimento = np.random.choice(['positivo', 'negativo', 'incerto'], p=[0.6, 0.3, 0.1])
            else:
                # Segunda metade: mais negativos
                sentimento = np.random.choice(['positivo', 'negativo', 'incerto'], p=[0.3, 0.6, 0.1])
            sentimentos.append(sentimento)
        
        df_tendencias = pd.DataFrame({
            'data': dates,
            'sentimento': sentimentos,
            'probabilidade': np.random.uniform(0.5, 0.95, len(dates))
        })
        
        # Gráfico de tendências
        fig = px.line(
            df_tendencias.groupby(['data', 'sentimento']).size().reset_index(name='count'),
            x='data',
            y='count',
            color='sentimento',
            title='Tendência de Sentimentos ao Longo do Tempo'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas mensais
        df_tendencias['mes'] = df_tendencias['data'].dt.month
        stats_mensais = df_tendencias.groupby(['mes', 'sentimento']).size().unstack(fill_value=0)
        
        st.subheader("📊 Estatísticas Mensais")
        st.dataframe(stats_mensais)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🤖 Desenvolvido com:**")
    st.markdown("- Streamlit")
    st.markdown("- PySentimiento")
with col2:
    st.markdown("**📊 Funcionalidades:**")
    st.markdown("- Análise Individual")
    st.markdown("- Análise em Lote")
    st.markdown("- Dashboard Completo")
with col3:
    st.markdown("**🔧 Tecnologias:**")
    st.markdown("- Python")
    st.markdown("- NLP")
    st.markdown("- IA Generativa")

st.markdown("*Análise de Sentimento Avançada - Modelo Completo*")