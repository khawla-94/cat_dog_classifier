import streamlit as st
import tensorflow as tf
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('cat_dog_classifier_model.keras')
  return model

with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
# MSDE5 : Deep Learning Project
## Cat Vs Dog Classification using CNN and Transfer Learning
""")

st.write("Upload an image to classify whether it's a cat or a dog.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((200, 200))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = st.success(f"Prediction: Dog (Probability: {prediction[0][0]:.2f})")
    else:
        result = st.success(f"Prediction: Cat (Probability: {1 - prediction[0][0]:.2f})")

    # Display the uploaded image and the prediction result
    st.image(img, caption="Uploaded Image", use_column_width=True)
