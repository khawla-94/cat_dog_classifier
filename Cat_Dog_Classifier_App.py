import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tempfile

@st.cache(allow_output_mutation=True)
def load_model():
  model_url = "https://github.com/khawla-94/cat_dog_classifier/blob/main/cat_dog_classifier_model.tflite"
  model_content = requests.get(model_url).content
  # Save the file in a temporary file:
  with tempfile.NamedTemporaryFile(delete = False) as temp_model_file:
    temp_model_file.write(model_content)
  # Initialize the interpreter with a temporary file:
  model = tf.lite.Interpreter(model_path = temp_model_file.name)
  model.allocate_tensors()
  return model

st.write("""
# MSDE5 : Deep Learning Project
## Cat Vs Dog Classification using CNN and Transfer Learning
""")

st.write("Upload an image to classify whether it's a cat or a dog.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# make predictions:
def predict_image(img, model):
  img = Image.open(uploaded_file).convert('RGB')
  img = img.resize((200, 200))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)
  img_array /= 255.0

  input_tensor_index = model.get_input_details()[0]['index']
  output = model.tensor(model.get_output_details()[0]['index'])

  model.set_tensor(input_tensor_index, img_array)
  model.invoke()
  prediction = output()
  
  if prediction[0][0] > 0.5:
    result = st.success(f"Prediction: Dog (Probability: {prediction[0][0]:.2f})")
  else:
    result = st.success(f"Prediction: Cat (Probability: {1 - prediction[0][0]:.2f})")
  
# Loading the model:
  model = load_model()

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  # Display the uploaded image and the prediction result
  st.image(img, caption="Uploaded Image", use_column_width=True)

  result = predict_image(image, model)
  d
  if result == 'Dog':
    st.success("Prediction: Dog")
  else:
    st.success("Prediction: Cat")
