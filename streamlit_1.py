#from pyngrok import ngrok
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np
from tensorflow.keras.models import load_model




model = load_model('CNN_Model.h5')
st.title('bonjour')

SIZE =  1000

canvas_result = st_canvas(
    fill_color='#ffffff',
    stroke_width=10,
    stroke_color='#ffffff',
    background_color='#000000',
    height=300,width=500,
    drawing_mode='freedraw',
    key='canvas'

)

if canvas_result.image_data is not None :
  img = cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))
  img_rescaling = cv2.resize(img,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
  st.write('input image')
  st.image(img_rescaling)

if st.button('predict') :
  test_x = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  pred = model.predict(test_x.reshape(1,784))
  st.write(f'result: {np.argmax(pred[0])}')
  st.bar_chart(pred[0])


# st.header(f"Chiffre predit : {model.predict(sample).argmax()}")
# if st.button("Rerun"):
#     return
# return
