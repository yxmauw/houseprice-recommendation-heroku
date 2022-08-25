# https://www.analyticsvidhya.com/blog/2021/07/streamlit-quickly-turn-your-ml-models-into-web-apps/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.impute import KNNImputer
import pickle

import streamlit as st

from model_methods import predict

# configuration of the page
st.set_page_config(
    layout="centered", 
    page_icon="🏠", 
    page_title="Are you planning to sell your house?", 
    initial_sidebar_state='auto',
)

st.title("🏠Ames Housing Sale Price recommendation tool")
st.markdown('''
The algorithm driving this app is built on
historical housing sale price data to generate
recommended Sale Price!
''')
###########################################################
def predict_price():
    data = list(map(float, [gr_liv_area,
                            (float(gr_liv_area))**2,
                            (float(gr_liv_area))**3,
                            overall_qual, 
                            total_bsmt_sf,
                            garage_area,
                            year_built,
                            mas_vnr_area]))
    result = np.format_float_positional((predict(data)[0]), unique=False, precision=0)
    st.info(f'# Our SalePrice suggestion is ${result}')
    st.write('with an estimated uncertainty of ± \$11K')

st.markdown('''
Please enter your house details to get a 
Sale Price suggestion 🙂
''')
gr_liv_area = st.text_input('Enter house ground living area in square feet', '')
overall_qual = np.nan
total_bsmt_sf = st.text_input('Enter house total basement area in square feet', '')
garage_area = st.text_input('Enter house garage area in square feet', '')
year_built = st.text_input('Enter the year your house was built', '')
mas_vnr_area = st.text_input('Enter house masonry veneer area in square feet', '')

if st.button('Submit'):
    with st.sidebar:
        try: 
            predict_price()
        except:
            st.warning('''Oops, looks like you missed a spot. 
            Please complete all fields to get a quote estimate 
            for property Sale Price 🙏. 
            \n\n Thank you. 🙂''')
