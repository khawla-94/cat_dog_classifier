# Importing dependencies:
import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import tempfile

# Loading the model:
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://github.com/khawla-94/cat_dog_classifier/raw/main/cat_dog_classifier_model.tflite"
    model_content = requests.get(model_url).content

    with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
        temp_model_file.write(model_content)

    model = tf.lite.Interpreter(model_path=temp_model_file.name)
    model.allocate_tensors()
    return model

# Make predictions:
def predict_image(img_to_predict, model):
  
    img = Image.open(img_to_predict).convert('RGB')
    img = img.resize((200, 200))
    img_array =  tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array /= 255.0

    input_tensor_index = model.get_input_details()[0]['index']
    output = model.tensor(model.get_output_details()[0]['index'])

    model.set_tensor(input_tensor_index, img_array)
    model.invoke()
    prediction = output()

    if prediction[0, 0] > 0.5:
        return 'Dog'
    else:
        return 'Cat'

# Loading the model:
model = load_model()

# Streamlit Web App:
st.write("""
# MSDE5 : Deep Learning Project
## Cat Vs Dog Classification using CNN and Transfer Learning updated
""")

st.sidebar.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EvMbMNRHm_aOf1n4tDO1Xg.jpeg", width=250)
st.sidebar.write("This is a classification model of cat and dog images")
# st.markdown("This project was made by : **KHAWLA BADDAR**")
st.write("Upload an image to classify whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    result = predict_image(image, model)
    
    # Display the image:
    if result == 'Dog':
        st.success("Prediction: It's a Dog ")
    else:
        st.success(f"Prediction: It's a Cat ")
