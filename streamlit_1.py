import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



test = pd.read_csv('data/test_cnn.csv')
model = load_model('CNN_Model.h5')

# //////INTRO//////

st.title("DEPLOIEMENT MODEL CNN")

button = st.sidebar.selectbox("",("exo_1", "exo_2_canvas"))
col1,col2,col3,col4,col5 = st.columns(5)



sample = test.sample(1)
sample = sample.values.reshape(-1,28,28,1)/ 255.0

def first_pred(test):

    #with col1:
    st.image(sample, use_column_width="always")
    #with col2:
    st.button('Predict')
    st.header(f"Chiffre predit : {model.predict(sample).argmax()}")

# ////// EXO 1 //////

if button == "exo_1":


    first_pred(test)


# ////// EXO 2 //////

if button == "exo_2_canvas":




# /////CANVAS //////
    #with col1:
    SIZE =  800
    st.text('dessine un chiffre')
    canvas_result = st_canvas(
    fill_color='#ffffff',
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 30, 9),
    stroke_color='#ffffff',
    background_color='#000000',
    height=350,width=350,
    drawing_mode='freedraw',
    key='canvas'
    )


    #with col2:
    if canvas_result.image_data is not None :
        img = cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))
        img_rescaling = cv2.resize(img,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
        st.write('input image')
        st.image(img_rescaling)


        #input_numpy_array = np.array(canvas_result.image_data)

    if st.button('predict') :
        test_x = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        pred = model.predict(test_x.reshape(-1,28,28,1))
        st.write(f'result: {np.argmax(pred[0])}')
        st.bar_chart(pred[0])
