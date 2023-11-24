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
  
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((200, 200))
    img_array =  tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array /= 255.0

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

st.sidebar.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EvMbMNRHm_aOf1n4tDO1Xg.jpeg", width=250)

st.write("Upload an image to classify whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Faire une prédiction avec le modèle
    result = predict_image(image, model)
    
    # Afficher la prédiction en mettant en évidence le résultat
    if result == 'Dog':
        st.success("Prediction: It's a Dog ")
    else:
        st.success(f"Prediction: It's a Cat ")

# # Import Dependencies:
# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import requests
# from io import BytesIO

# # Loading the model:
# @st.cache(allow_output_mutation = True)
# def load_model():
#     model_url = "https://github.com/khawla-94/cat_dog_classifier/raw/main/cat_dog_classifier_model.tflite" 
#     model_content = requests.get(model_url).content
#     model = tf.lite.Interpreter(model_content = model_content)
#     model.allocate_tensors()
#     return model

# # Get input and output details:
# interpreter = load_model()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Define the image size expected by the model:
# input_shape = input_details[0]['shape'][1:3]

# # PreProcess the model:

# def preprocess_image(image_path):
#     try:
#         img = Image.open(image_path).convert('RGB')
#         img = img.resize(input_shape, Image.BICUBIC)
#         img_array = np.array(img, dtype = np.float32) / 255.0
#         img_array = np.expand_dims(img_array, axis = 0)
#         return img_array
#     except Exception as e:
#         st.error(f"Error preprocessing image: {e}")
#         return None

# # Function to make predictions:

# def predict_image(image_path):
#     input_data = preprocess_image(image_path)
#     if input_data is None:
#         return None
#     try:
#         print("Input data shape:", input_data.shape if hasattr(input_data, 'shape') else "Not available")
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])
#         return output_data[0][0]
#     except Exception as e:
#         st.error(f"Error setting tensor: {e}")
#         return None

# # Streamlit App:
# st.write("""
# # MSDE5 : Deep Learning Project
# ## Cat Vs Dog Classification using CNN and Transfer Learning
# """)
# st.sidebar.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EvMbMNRHm_aOf1n4tDO1Xg.jpeg", width=250)

# st.write("Upload an image to classify whether it's a cat or a dog.")

# uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

# # Display the uploaded file:

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption = "Uploaded Image", use_column_width = True)
#     # make predictions:
#     prediction = predict_image(uploaded_file)
#     # Display the prediction:
#     if prediction > 0.5:
#         st.success("Prediction: It's a Dog ")
#     else:
#         st.success("Prediction: It's a Cat")
