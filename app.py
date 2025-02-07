import streamlit as st
import numpy as np
import joblib

@st.cache_resource

def load_model():
    model=joblib.load('RF_Model.pkl')
    return model

st.cache_resource.clear()

st.title('GENDER CLASSIFICATION APP')
st.subheader('This app categorizes an individual as male or female depending on their features')

model=load_model()

if model:
    st.header('Please enter your details')

long_hair=st.selectbox('LONG HAIR?',options=[(0,'No'),(1,'Yes')],format_func=lambda x: x[0])
long_hair_values=long_hair[0]

forehead_width_cm=st.number_input('FOREHEAD WIDTH IN CM',value=0)

forehead_height_cm=st.number_input('FOREHEAD HEIGHT IN CM',value=0)

nose_wide=st.selectbox('WIDE NOSE',options=[(0,'No'),(1,'Yes')],format_func=lambda x: x[0])
nose_wide_values=nose_wide[0]

nose_long=st.selectbox('LONG NOSE',options=[(0,'No'),(1,'Yes')],format_func=lambda x: x[0])
nose_long_values=nose_long[0]

lips_thin=st.selectbox('THIN LIPS',options=[(0,'No'),(1,'Yes')],format_func=lambda x: x[0])
lips_thin_values=lips_thin[0]

distance_nose_to_lip_long=st.selectbox('LONG DISTANCE BETWEEN NOSE AND LIPS',options=[(0,'No'),(1,'Yes')],format_func=lambda x: x[0])
distance_nose_to_lip_long_values=distance_nose_to_lip_long[0]

input_data=np.array([long_hair_values,forehead_width_cm,forehead_height_cm,nose_wide_values,nose_long_values,lips_thin_values,distance_nose_to_lip_long_values])

if st.button('Predict Gender',key='predict_button'):
    prediction=model.predict(input_data.reshape(1,-1)) 
    gender_prediction = 'Yes' if prediction[0] == 1 else 'No'
    st.subheader(f'Predicted Gender:{gender_prediction}')

st.write('We are all beautiful')





