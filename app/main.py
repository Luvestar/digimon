import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Digimon Evolution Classifier",
    page_icon="🦖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🦖 Digimon Evolution Level Classifier")
st.markdown("""
### Predice el nivel evolutivo de un Digimon basado en sus estadísticas
Este modelo de Machine Learning clasifica Digimon en 8 niveles evolutivos:
**Baby, In-Training, Rookie, Champion, Ultimate, Mega, Ultra, Armor**
""")

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/raw/DigiDB_digimonlist.csv')
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.info("Asegúrate de que el archivo data/raw/DigiDB_digimonlist.csv existe")
        return None

# Sidebar para navegación
st.sidebar.title("🎮 Navegación")
page = st.sidebar.radio(
    "Ir a:",
    ["🏠 Dashboard", "📊 Análisis de Datos", "🤖 Entrenar Modelo", "🎮 Clasificador"],
    key="navigation"
)

# Cargar datos una vez
df = load_data()

# ========== DASHBOARD ==========
if page == "🏠 Dashboard":
    st.header("🏠 Dashboard")
    
    if df is None:
        st.error("No se pudo cargar el dataset")
        st.stop()
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Digimon", len(df))
    with col2:
        st.metric("Niveles Evolutivos", df['Stage'].nunique())
    with col3:
        st.metric("Tipos", df['Type'].nunique())
    with col4:
        st.metric("Atributos", df['Attribute'].nunique())
    
    st.markdown("---")
    
    # Distribución de Stages
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Distribución por Nivel Evolutivo")
        stage_counts = df['Stage'].value_counts().reset_index()
        stage_counts.columns = ['Stage', 'Count']
        
        fig = px.bar(
            stage_counts, 
            x='Stage', 
            y='Count',
            color='Stage',
            title="Cantidad de Digimon por Nivel Evolutivo"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Objetivo del Modelo")
        st.info("""
        **Variable objetivo:** Stage  
        **Clases:** 8 niveles  
        **Tipo:** Clasificación multiclase  
        **Algoritmo:** Random Forest
        """)
    
    # Preview de datos
    st.subheader("👀 Vista previa de los datos")
    st.dataframe(df.head(10), use_container_width=True)

# ========== ANÁLISIS DE DATOS ==========
elif page == "📊 Análisis de Datos":
    st.header("📊 Análisis Exploratorio de Datos")
    
    if df is None:
        st.error("No se pudo cargar el dataset")
        st.stop()
    
    # Filtros
    st.sidebar.subheader("🔍 Filtros")
    selected_stages = st.sidebar.multiselect(
        "Niveles Evolutivos",
        options=df['Stage'].unique(),
        default=df['Stage'].unique()
    )
    
    # Filtrar datos
    filtered_df = df[df['Stage'].isin(selected_stages)]
    
    # Pestañas
    tab1, tab2, tab3 = st.tabs(["📊 Distribuciones", "🔥 Correlaciones", "📈 Comparaciones"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución de Estadísticas")
            stat_to_plot = st.selectbox(
                "Seleccionar estadística",
                ['Lv 50 HP', 'Lv50 SP', 'Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd'],
                key="stat_select"
            )
            
            fig = px.histogram(
                filtered_df, 
                x=stat_to_plot,
                color='Stage',
                nbins=30,
                title=f"Distribución de {stat_to_plot}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Boxplot por Stage")
            fig = px.box(
                filtered_df, 
                x='Stage', 
                y='Lv50 Atk',
                color='Stage',
                title="Distribución de Ataque por Nivel Evolutivo"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔗 Matriz de Correlación")
        
        numeric_cols = ['Lv 50 HP', 'Lv50 SP', 'Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd']
        numeric_df = filtered_df[numeric_cols]
        
        # Renombrar columnas
        numeric_df.columns = ['HP', 'SP', 'Atk', 'Def', 'Int', 'Spd']
        
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}'
        ))
        
        fig.update_layout(title="Correlación entre Estadísticas", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Comparación de Atributos")
        
        fig = px.scatter(
            filtered_df,
            x='Lv50 Atk',
            y='Lv50 Def',
            color='Stage',
            size='Lv 50 HP',
            hover_name='Digimon',
            title="Ataque vs Defensa por Stage"
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== ENTRENAR MODELO ==========
elif page == "🤖 Entrenar Modelo":
    st.header("🤖 Entrenamiento del Modelo")
    
    if df is None:
        st.error("No se pudo cargar el dataset")
        st.stop()
    
    # Sidebar configuración
    st.sidebar.subheader("⚙️ Configuración del Modelo")
    test_size = st.sidebar.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
    n_estimators = st.sidebar.slider("Número de árboles", 50, 500, 100)
    max_depth = st.sidebar.slider("Profundidad máxima", 3, 20, 10)
    
    # Preprocesar datos
    st.subheader("🔄 Preprocesamiento de Datos")
    
    if st.button("🔧 Ejecutar Preprocesamiento", key="preprocess_btn"):
        with st.spinner("Preprocesando datos..."):
            # Crear copia
            df_processed = df.copy()
            
            # One-Hot Encoding para Type
            type_dummies = pd.get_dummies(df_processed['Type'], prefix='Type')
            df_processed = pd.concat([df_processed, type_dummies], axis=1)
            
            # Label Encoding para Attribute
            le_attribute = LabelEncoder()
            df_processed['Attribute_encoded'] = le_attribute.fit_transform(df_processed['Attribute'])
            
            # Definir características
            feature_cols = [
                'Type_Vaccine', 'Type_Virus', 'Type_Data', 'Type_Free',
                'Attribute_encoded',
                'Lv 50 HP', 'Lv50 SP', 'Lv50 Atk', 
                'Lv50 Def', 'Lv50 Int', 'Lv50 Spd'
            ]
            
            X = df_processed[feature_cols]
            y = df_processed['Stage']
            
            # Normalizar
            scaler = StandardScaler()
            numeric_cols = ['Lv 50 HP', 'Lv50 SP', 'Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd']
            X.loc[:, numeric_cols] = scaler.fit_transform(X[numeric_cols])
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Guardar en session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.le_attribute = le_attribute
            st.session_state.feature_cols = feature_cols
            st.session_state.preprocessed = True
            
            st.success(f"✅ Datos preprocesados! Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Entrenar modelo
    st.subheader("🚀 Entrenamiento del Modelo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎯 Entrenar Modelo", disabled='preprocessed' not in st.session_state, key="train_btn"):
            with st.spinner("Entrenando modelo..."):
                # Crear y entrenar modelo
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    class_weight='balanced'
                )
                
                model.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Evaluar
                y_pred = model.predict(st.session_state.X_test)
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                
                # Guardar en session state
                st.session_state.model = model
                st.session_state.trained = True
                st.session_state.accuracy = accuracy
                
                # Guardar archivos
                os.makedirs('models', exist_ok=True)
                joblib.dump(model, 'models/random_forest.pkl')
                joblib.dump(st.session_state.scaler, 'models/scaler.pkl')
                joblib.dump(st.session_state.le_attribute, 'models/attribute_encoder.pkl')
                
                st.success(f"✅ Modelo entrenado! Precisión: {accuracy:.2%}")
    
    with col2:
        if 'trained' in st.session_state:
            st.subheader("📊 Métricas")
            st.info(f"""
            **Precisión:** {st.session_state.accuracy:.2%}  
            **Muestras entrenamiento:** {len(st.session_state.X_train)}  
            **Muestras prueba:** {len(st.session_state.X_test)}
            """)
    
    # Evaluación del modelo
    if 'trained' in st.session_state:
        st.subheader("📈 Evaluación del Modelo")
        
        tab1, tab2 = st.tabs(["Matriz de Confusión", "Importancia de Características"])
        
        with tab1:
            # Matriz de confusión
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            cm = confusion_matrix(st.session_state.y_test, y_pred)
            classes = sorted(st.session_state.y_test.unique())
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=classes,
                y=classes,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}'
            ))
            
            fig.update_layout(
                title="Matriz de Confusión",
                xaxis_title="Predicción",
                yaxis_title="Real",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Importancia de características
            if hasattr(st.session_state.model, 'feature_importances_'):
                importances = st.session_state.model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': st.session_state.feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Características Más Importantes"
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ========== CLASIFICADOR ==========
elif page == "🎮 Clasificador":
    st.header("🎮 Clasificador Interactivo")
    
    # Verificar si el modelo está entrenado
    model_path = 'models/random_forest.pkl'
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load('models/scaler.pkl')
            le_attribute = joblib.load('models/attribute_encoder.pkl')
            model_loaded = True
        except Exception as e:
            model_loaded = False
            st.warning(f"⚠️ Error cargando el modelo: {e}")
    else:
        model_loaded = False
        st.warning("⚠️ Modelo no encontrado. Ve a 'Entrenar Modelo' para crear uno.")
    
    # Layout en dos columnas
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuración del Digimon")
        
        digimon_name = st.text_input("Nombre del Digimon", value="MiDigimon")
        
        col_type, col_attr = st.columns(2)
        with col_type:
            digimon_type = st.selectbox(
                "Tipo",
                ["Vaccine", "Virus", "Data", "Free"],
                index=0
            )
        
        with col_attr:
            digimon_attribute = st.selectbox(
                "Atributo",
                ["Neutral", "Fire", "Water", "Plant", "Electric", 
                 "Earth", "Wind", "Light", "Dark"],
                index=0
            )
        
        # Estadísticas
        st.markdown("**📊 Estadísticas (Nivel 50)**")
        
        hp = st.slider("HP", 500, 2500, 1000, key="hp_slider")
        sp = st.slider("SP", 50, 250, 100, key="sp_slider")
        attack = st.slider("Ataque", 50, 250, 100, key="atk_slider")
        defense = st.slider("Defensa", 50, 250, 100, key="def_slider")
        intelligence = st.slider("Inteligencia", 50, 250, 100, key="int_slider")
        speed = st.slider("Velocidad", 50, 250, 100, key="spd_slider")
        
        # Botón para predecir
        predict_button = st.button("🎯 Predecir Nivel Evolutivo", 
                                  type="primary", 
                                  disabled=not model_loaded,
                                  use_container_width=True,
                                  key="predict_button")
        
        # Ejemplos si hay datos
        if df is not None and not df.empty:
            st.markdown("**🔥 Ejemplos Rápidos**")
            
            # Obtener lista de Digimon únicos
            digimon_list = [""] + sorted(df['Digimon'].unique().tolist())
            
            selected_digimon = st.selectbox(
                "Seleccionar Digimon de ejemplo",
                options=digimon_list,
                index=0,
                key="example_select"
            )
            
            if selected_digimon and selected_digimon != "":
                if st.button("📥 Cargar ejemplo", key="load_example"):
                    try:
                        # Buscar el Digimon seleccionado
                        example_row = df[df['Digimon'] == selected_digimon]
                        
                        if not example_row.empty:
                            example_data = example_row.iloc[0]
                            
                            # Mostrar información
                            st.success(f"✅ Ejemplo cargado: {selected_digimon}")
                            st.info(f"""
                            **Tipo:** {example_data['Type']}  
                            **Atributo:** {example_data['Attribute']}  
                            **Stage:** {example_data['Stage']}  
                            **HP:** {example_data['Lv 50 HP']}  
                            **Ataque:** {example_data['Lv50 Atk']}  
                            **Defensa:** {example_data['Lv50 Def']}
                            """)
                    except Exception as e:
                        st.error(f"Error cargando ejemplo: {e}")
    
    with col2:
        st.subheader("📊 Resultados de Predicción")
        
        # Mostrar instrucciones si no hay predicción
        if not model_loaded:
            st.info("""
            ### ⚠️ Primero entrena el modelo
            1. Ve a **"🤖 Entrenar Modelo"**
            2. Haz clic en **"Ejecutar Preprocesamiento"**
            3. Haz clic en **"Entrenar Modelo"**
            4. Regresa aquí para usar el clasificador
            """)
        else:
            st.info("""
            ### 👈 Configura tu Digimon
            Usa los controles de la izquierda para:
            1. Seleccionar tipo y atributo
            2. Ajustar las estadísticas
            3. Presionar **"Predecir Nivel Evolutivo"**
            
            También puedes seleccionar un Digimon de ejemplo.
            """)
        
        # Procesar predicción cuando se presiona el botón
        if predict_button and model_loaded:
            with st.spinner("Analizando Digimon..."):
                try:
                    # Preparar datos de entrada
                    # One-Hot Encoding para Type
                    type_dummies = pd.DataFrame({
                        'Type_Vaccine': [1 if digimon_type == 'Vaccine' else 0],
                        'Type_Virus': [1 if digimon_type == 'Virus' else 0],
                        'Type_Data': [1 if digimon_type == 'Data' else 0],
                        'Type_Free': [1 if digimon_type == 'Free' else 0]
                    })
                    
                    # Codificar Attribute
                    try:
                        attribute_encoded = le_attribute.transform([digimon_attribute])[0]
                    except ValueError:
                        # Si el atributo no está en el encoder, usar el más común
                        attribute_encoded = le_attribute.transform(['Neutral'])[0]
                    
                    # Valores numéricos
                    numeric_values = np.array([[
                        hp, sp, attack, defense, intelligence, speed
                    ]])
                    
                    # Normalizar
                    numeric_scaled = scaler.transform(numeric_values)[0]
                    
                    # Combinar características
                    features = np.concatenate([
                        type_dummies.values[0],
                        [attribute_encoded],
                        numeric_scaled
                    ]).reshape(1, -1)
                    
                    # Hacer predicción
                    prediction = model.predict(features)[0]
                    
                    # Obtener probabilidades
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features)[0]
                        classes = model.classes_
                        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
                    else:
                        prob_dict = {prediction: 1.0}
                    
                    # Guardar en session state
                    st.session_state.prediction = prediction
                    st.session_state.prob_dict = prob_dict
                    st.session_state.input_stats = {
                        'HP': hp, 'SP': sp, 'Atk': attack, 'Def': defense,
                        'Int': intelligence, 'Spd': speed
                    }
                    st.session_state.current_type = digimon_type
                    st.session_state.current_attribute = digimon_attribute
                    
                except Exception as e:
                    st.error(f"❌ Error en la predicción: {e}")
        
        # Mostrar resultados si hay predicción
        if 'prediction' in st.session_state:
            # Mostrar predicción principal
            st.markdown("### 🎯 Predicción")
            
            # Emojis por stage
            stage_emojis = {
                'Baby': '👶',
                'In-Training': '🌱',
                'Rookie': '🦊',
                'Champion': '🦁',
                'Ultimate': '🐉',
                'Mega': '👑',
                'Ultra': '🌟',
                'Armor': '🛡️'
            }
            
            emoji = stage_emojis.get(st.session_state.prediction, '🦖')
            
            col_pred, col_conf = st.columns([2, 1])
            
            with col_pred:
                current_type_display = st.session_state.get('current_type', digimon_type)
                current_attr_display = st.session_state.get('current_attribute', digimon_attribute)
                
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; border-radius: 10px; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white;'>
                    <h1 style='font-size: 48px; margin: 0;'>{emoji}</h1>
                    <h2 style='margin: 10px 0;'>{st.session_state.prediction}</h2>
                    <h3 style='margin: 0;'>{current_type_display} - {current_attr_display}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col_conf:
                if 'prob_dict' in st.session_state:
                    max_prob = max(st.session_state.prob_dict.values())
                    st.metric("Confianza", f"{max_prob:.1%}")
            
            st.markdown("---")
            
            # Gráfico de probabilidades
            if 'prob_dict' in st.session_state:
                st.subheader("📈 Probabilidades por Nivel Evolutivo")
                
                prob_df = pd.DataFrame({
                    'Stage': list(st.session_state.prob_dict.keys()),
                    'Probability': list(st.session_state.prob_dict.values())
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(
                    prob_df,
                    x='Stage',
                    y='Probability',
                    color='Probability',
                    color_continuous_scale='Viridis',
                    title="Distribución de Probabilidades"
                )
                
                fig.update_traces(
                    texttemplate='%{y:.1%}',
                    textposition='outside'
                )
                
                fig.update_layout(
                    yaxis_title="Probabilidad",
                    yaxis=dict(tickformat='.0%'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Información")
st.sidebar.info("""
**Dataset:** Digimon Story Cyber Sleuth  
**Modelo:** Random Forest Classifier  
**Características:** 11 variables  
**Clases:** 8 niveles evolutivos
""")