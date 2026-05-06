import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")
st.title("🌸 EndémicaEns: Flora de Ensenada")

# --- DICCIONARIO DE ESPECIES ---
# Nota: Corregido "rosa de castilla" con la letra 'i'
especies_info = {
    "amapola_de_california": {
        "nombre": "Amapola de California",
        "cientifico": "Eschscholzia californica",
        "estado": "Nativa",
        "info": "Flor brillante de color naranja. Es muy resistente a la sequía.",
        "cuidados": "Mucho sol y casi nada de agua.",
        "plantacion": "Finales de invierno."
    },
    "choya_californiana": {
        "nombre": "Choya / Cacto de Ensenada",
        "cientifico": "Cylindropuntia californica",
        "estado": "Nativa",
        "info": "Cacto muy espinoso típico del matorral costero.",
        "cuidados": "Cero riego adicional, suelo muy arenoso.",
        "plantacion": "Todo el año."
    },
    "encelia farinosa": {
        "nombre": "Incienso / Encelia",
        "cientifico": "Encelia farinosa",
        "estado": "Nativa",
        "info": "Arbusto de flores amarillas. Sus hojas son grisáceas.",
        "cuidados": "Muy poca agua, pleno sol.",
        "plantacion": "Primavera."
    },
    "encino_quercus_agrifolia": {
        "nombre": "Encino Californiano",
        "cientifico": "Quercus agrifolia",
        "estado": "Nativa / Protegida",
        "info": "Árbol majestuoso fundamental para el ecosistema local.",
        "cuidados": "Evitar exceso de riego en el tronco.",
        "plantacion": "Otoño."
    },
    "lila_california_ceanothus": {
        "nombre": "Lila de California",
        "cientifico": "Ceanothus spp.",
        "estado": "Nativa Regional",
        "info": "Famosa por sus racimos de flores azules o moradas.",
        "cuidados": "Requiere poco riego una vez establecida.",
        "plantacion": "Otoño o Invierno."
    },
    "maguey de la costa_agabe_shawii": { 
        "nombre": "Maguey de Costa",
        "cientifico": "Agave shawii",
        "estado": "Nativa Regional",
        "info": "Suculenta protegida que crece frente al mar.",
        "cuidados": "Suelo arenoso y mucha brisa marina.",
        "plantacion": "Invierno."
    },
    "rosa de castilla_rosa_minutifolia": {
        "nombre": "Rosa de Castilla",
        "cientifico": "Rosa minutifolia",
        "estado": "Endémica de BC",
        "info": "La joya de Ensenada. Pequeña, espinosa y flores rosas vibrantes.",
        "cuidados": "Cero riego una vez establecida.",
        "plantacion": "Temporada de lluvias."
    },
    "salvia de munz_salvia_munzii": {
        "nombre": "Salvia de Munz",
        "cientifico": "Salvia munzii",
        "estado": "Nativa Regional",
        "info": "Arbusto aromático esencial para polinizadores.",
        "cuidados": "Sol directo y suelo bien drenado.",
        "plantacion": "Primavera."
    }
}

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('modelo_flores_nuevo.keras')
    except:
        return tf.keras.models.load_model('modelo_flores_nuevo.keras', compile=False)

model = load_model()

# --- INTERFAZ ---
archivo = st.file_uploader("Sube una foto de la flora local", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    img = Image.open(archivo)
    st.image(img, use_container_width=True)
    
    # Preprocesamiento
    img_resized = img.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    # Predicción
    with st.spinner('Identificando...'):
        pred = model.predict(img_array)
        score = tf.nn.softmax(pred[0])
    
    # LISTA MANUAL: Debe coincidir al 100% con el orden alfabético de tus carpetas
    nombres_carpetas = [
        "amapola_de_california",
        "choya_californiana",
        "encelia farinosa",
        "encino_quercus_agrifolia",
        "lila_california_ceanothus",
        "maguey de la costa_agabe_shawii",
        "rosa de castilla_rosa_minutifolia",
        "salvia de munz_salvia_munzii"
    ]
    
    indice_detectado = np.argmax(score)
    clase_detectada = nombres_carpetas[indice_detectado]
    confianza = 100 * np.max(score)

    # Mostrar resultados
    info = especies_info[clase_detectada]
    st.success(f"## {info['nombre']}")
    st.write(f"**Confianza:** {confianza:.2f}%")
    
    with st.expander("📖 Ver Detalles Técnicos"):
        st.write(f"**Científico:** *{info['cientifico']}*")
        st.write(f"**Estado:** {info['estado']}")
        st.info(info['info'])
        st.warning(f"**Cuidados:** {info['cuidados']}")
