import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")
st.title("🌸 EndémicaEns: Flora de Ensenada")

# --- DICCIONARIO DE ESPECIES ---
# El orden de estas llaves determinará el orden alfabético que usa la IA
especies_info = {
    "amapola_de_california": {
        "nombre": "Amapola_de_California",
        "cientifico": "Eschscholzia californica",
        "estado": "Nativa",
        "info": "Flor brillante de color naranja. Es muy resistente a la sequía y se cierra por las noches.",
        "cuidados": "Mucho sol y casi nada de agua.",
        "plantacion": "Finales de invierno."
    },
    "choya_californiana": {
        "nombre": "Choya / Cacto de Ensenada",
        "cientifico": "Cylindropuntia californica",
        "estado": "Nativa",
        "info": "Cacto muy espinoso típico del matorral costero. Sus espinas se pegan fácilmente.",
        "cuidados": "Cero riego adicional, suelo muy arenoso.",
        "plantacion": "Todo el año."
    },
    "encelia farinosa": {
        "nombre": "Incienso / Encelia",
        "cientifico": "Encelia farinosa",
        "estado": "Nativa",
        "info": "Arbusto de flores amarillas. Sus hojas grisáceas reflejan la luz para sobrevivir al calor.",
        "cuidados": "Muy poca agua, pleno sol.",
        "plantacion": "Primavera."
    },
    "encino_quercus_agrifolia": {
        "nombre": "Encino Californiano",
        "cientifico": "Quercus agrifolia",
        "estado": "Nativa / Protegida",
        "info": "Árbol majestuoso de los arroyos de Ensenada. Fundamental para el ecosistema local.",
        "cuidados": "Evitar exceso de riego en el tronco durante el verano.",
        "plantacion": "Otoño."
    },
    "lila_california_ceanothus": {
        "nombre": "Lila de California",
        "cientifico": "Ceanothus spp.",
        "estado": "Nativa Regional",
        "info": "Famosa por sus racimos de flores azules o moradas. Atrae a muchas mariposas.",
        "cuidados": "Requiere poco riego una vez establecida.",
        "plantacion": "Otoño o Invierno."
    },
    "maguey de la costa_agabe_shawii": { 
        "nombre": "Maguey de Costa",
        "cientifico": "Agave shawii",
        "estado": "Nativa Regional",
        "info": "Suculenta protegida que crece frente al mar. Sus flores atraen a murciélagos.",
        "cuidados": "Suelo arenoso y mucha brisa marina.",
        "plantacion": "Invierno."
    },
    "rosa de castilla_rosa_minutifolia": {
        "nombre": "Rosa de Castilla",
        "cientifico": "Rosa minutifolia",
        "estado": "Endémica de BC",
        "info": "La joya de Ensenada. Pequeña, muy espinosa y con flores rosas vibrantes.",
        "cuidados": "Cero riego una vez establecida.",
        "plantacion": "Temporada de lluvias."
    },
    "salvia de munz_salvia_munzii": {
        "nombre": "Salvia de Munz",
        "cientifico": "Salvia munzii",
        "estado": "Nativa Regional",
        "info": "Arbusto aromático de flores moradas, esencial para los polinizadores.",
        "cuidados": "Sol directo y suelo bien drenado.",
        "plantacion": "Primavera."
    }
}

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    try:
        # Intento de carga estándar
        return tf.keras.models.load_model('modelo_flores_nuevo.keras')
    except:
        # Carga sin compilación (por si hay diferencias de versiones)
        return tf.keras.models.load_model('modelo_flores_nuevo.keras', compile=False)

model = load_model()

# --- INTERFAZ DE USUARIO ---
archivo = st.file_uploader("Sube una foto de la flora local", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    img = Image.open(archivo)
    st.image(img, use_container_width=True) # Actualizado para versiones modernas de Streamlit
    
    # Preprocesamiento
    img_resized = img.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Crear el batch (1, 180, 180, 3)
    
    # Predicción
    with st.spinner('Identificando especie...'):
        pred = model.predict(img_array)
        score = tf.nn.softmax(pred[0])
    
    # Obtener etiquetas ordenadas alfabéticamente (igual que en el entrenamiento)
    nombres_carpetas = sorted(list(especies_info.keys()))
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
        st.info(f"**Descripción:** {info['info']}")
        st.warning(f"**Recomendación de Cuidado:** {info['cuidados']}")
        st.write(f"**Época ideal de plantación:** {info['plantacion']}")
