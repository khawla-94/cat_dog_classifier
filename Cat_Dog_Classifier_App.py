import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import tempfile

# Fonction pour charger le modèle depuis GitHub
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://github.com/khawla-94/cat_dog_classifier/raw/main/cat_dog_classifier_model.tflite"
    model_content = requests.get(model_url).content

    # Enregistrer le contenu dans un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
        temp_model_file.write(model_content)

    # Initialiser l'interpréteur avec le fichier temporaire
    model = tf.lite.Interpreter(model_path=temp_model_file.name)
    model.allocate_tensors()
    return model

# Fonction pour faire une prédiction
def predict_image(img, model):
  
    img = img.resize((200, 200))

    img_array =  np.array(img).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Préparer les données pour le modèle
    input_tensor_index = model.get_input_details()[0]['index']
    output = model.tensor(model.get_output_details()[0]['index'])

    # Faire une prédiction
    model.set_tensor(input_tensor_index, img_array)
    model.invoke()
    prediction = output()

    if prediction[0, 0] > 0.5:
        return 'Dog'
    else:
        return 'Cat'

# Chargement du modèle
model = load_model()

# Interface utilisateur Streamlit
st.write("""
# MSDE5 : Deep Learning Project
## Cat Vs Dog Classification using CNN and Transfer Learning
""")
st.write("Upload an image to classify whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Faire une prédiction avec le modèle
    result = predict_image(image, model)
    
    # Afficher la prédiction en mettant en évidence le résultat
    if result == 'Dog':
        st.success("Prediction: Dog 🐶")
    else:
        st.success("Prediction: Cat 🐱")
