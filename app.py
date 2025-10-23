import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image
from keras.models import load_model
import platform

# 🌸 --- Estilos femeninos y creativos ---
st.markdown("""
    <style>
        /* Fondo con degradado suave */
        .stApp {
            background: linear-gradient(180deg, #f6ecff 0%, #d8efff 100%);
            font-family: 'Poppins', sans-serif;
            color: #5a4a78;
        }

        /* Título principal */
        h1 {
            color: #b07acb;
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 0.3em;
            text-shadow: 1px 1px 4px rgba(176, 122, 203, 0.2);
        }

        /* Subtítulos */
        h2, h3 {
            color: #7b6dce;
            font-weight: 600;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f7e9ff;
            color: #5a4a78;
            border-radius: 20px;
            border: 1px solid #e3c7ff;
        }

        /* Botones principales */
        button[kind="primary"] {
            background-color: #a4c6f8 !important;
            color: white !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            border: none !important;
        }
        button[kind="primary"]:hover {
            background-color: #8ab5f5 !important;
        }

        /* Imagen decorativa */
        [data-testid="stImage"] img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 20px;
            box-shadow: 0 0 20px rgba(170, 120, 200, 0.3);
        }

        /* Cuadro de resultado */
        .resultado {
            background-color: #f3eaff;
            border-radius: 15px;
            padding: 15px;
            margin-top: 20px;
            border: 1px solid #d5c2ff;
            color: #5a4a78;
            text-align: center;
            font-size: 1.1em;
            box-shadow: 0 0 8px rgba(180, 140, 240, 0.2);
        }

        /* Texto general */
        .stMarkdown p {
            color: #4a3f60;
        }

        /* Encabezado resultado */
        h2.resultado-texto {
            color: #b07acb;
            text-align: center;
            font-weight: 700;
            margin-top: 20px;
        }

        /* Cámara */
        [data-testid="stCameraInput"] {
            background-color: #fff7ff;
            border-radius: 15px;
            border: 2px dashed #caa7f2;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# 🌷 --- Encabezado ---
st.title("💖 Reconocimiento de Imágenes Inteligente")
st.markdown("<p style='text-align:center;'>Analiza tus fotos y descubre qué detecta el modelo entrenado 🌸</p>", unsafe_allow_html=True)

# Mostrar versión de Python
st.write(f"Versión de Python: **{platform.python_version()}**")

# Cargar modelo entrenado
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Imagen decorativa
image = Image.open('camara.jpg')
st.image(image, width=350)

# 🌼 --- Panel lateral ---
with st.sidebar:
    st.subheader("✨ Acerca de esta app ✨")
    st.markdown("""
    Esta aplicación utiliza un modelo de **Teachable Machine**
    para reconocer gestos o direcciones a partir de una imagen capturada.  
    Usa la cámara para tomar una foto y ver el resultado 🌷
    """)

# 📸 --- Entrada de la cámara ---
img_file_buffer = st.camera_input("📷 Toma una foto con tu cámara")

# --- Procesamiento de imagen ---
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # 🔮 Predicción
    prediction = model.predict(data)
    st.markdown("<h2 class='resultado-texto'>💫 Resultado del reconocimiento:</h2>", unsafe_allow_html=True)

    if prediction[0][0] > 0.5:
        st.markdown(f"<div class='resultado'>🌸 Dirección detectada: <b>Izquierda</b><br>Probabilidad: {prediction[0][0]:.2f}</div>", unsafe_allow_html=True)
    elif prediction[0][1] > 0.5:
        st.markdown(f"<div class='resultado'>💜 Dirección detectada: <b>Arriba</b><br>Probabilidad: {prediction[0][1]:.2f}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='resultado'>🌈 No se detectó una dirección con alta confianza.</div>", unsafe_allow_html=True)
