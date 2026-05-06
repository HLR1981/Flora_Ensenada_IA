import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="EndémicaEns", page_icon="🌸")
st.title("🌸 EndémicaEns: Flora de Ensenada")

especies_info = {
    "amapola_de_california": {
        "nombre": "Amapola de California",
        "cientifico": "Eschscholzia californica",
        "info": "Flor brillante de color naranja, símbolo de la región.",
        "cuidados": "Sol directo y poca agua."
    },
    "choya_californiana": {"nombre": "Choya", "cientifico": "Cylindropuntia californica", "info": "Cacto muy espinoso.", "cuidados": "Cero riego."},
    "encelia farinosa": {"nombre": "Incienso / Encelia", "cientifico": "Encelia farinosa", "info": "Arbusto de flores amarillas.", "cuidados": "Poca agua."},
    "encino_quercus_agrifolia": {"nombre": "Encino Californiano", "cientifico": "Quercus agrifolia", "info": "Árbol majestuoso.", "cuidados": "No inundar el tronco."},
    "lila_california_ceanothus": {"nombre": "Lila de California", "cientifico": "Ceanothus spp.", "info": "Flores azules racimosas.", "cuidados": "Poco riego."},
    "maguey de la costa_agabe_shawii": {"nombre": "Maguey de Costa", "cientifico": "Agave shawii", "info": "Suculenta costera.", "cuidados": "Sol y brisa."},
    "rosa de castilla_rosa_minutifolia": {"nombre": "Rosa de Castilla", "cientifico": "Rosa minutifolia", "info": "Endémica y protegida.", "cuidados": "Cero riego."},
    "salvia de munz_salvia_munzii": {"nombre": "Salvia de Munz", "cientifico": "Salvia munzii", "info": "Aromática para abejas.", "cuidados": "Sol directo."}
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_v2.keras')

model = load_model()

archivo = st.file_uploader("Sube una foto", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo)
    st.image(img, use_container_width=True)
    
    img_resized = img.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    pred = model.predict(img_array)
    score = tf.nn.softmax(pred[0])
    
    # ESTA LISTA ES EL ORDEN ALFABÉTICO EXACTO (0 al 7)
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
    
    indice = np.argmax(score)
    clase = nombres_carpetas[indice]
    confianza = 100 * np.max(score)

    info = especies_info[clase]
    st.success(f"## {info['nombre']}")
    st.write(f"**Confianza:** {confianza:.2f}%")
    st.write(f"*Nombre científico:* {info['cientifico']}")

