import streamlit as st
from PIL import Image
import numpy as np
import keras
import cv2
#from main import teachable_machine_classification
st.title("Emotion Classification")
st.header("Sad Or Happy Classification")
st.text("Upload an Image for image classification as sad or happy")

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)
    img = cv2.resize(np.asarray(img), (256,256))
    # run the inference
    prediction = model.predict(np.expand_dims(img/255, 0))
    return prediction


uploaded_file = st.file_uploader("Choose a brain MRI ...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'imageclassifier.h5')
        if label > 0.5:
            st.write("People is/are sad")
        else:
            st.write("People is/are happy")