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

########################################################## 
ML_model = st.beta_container()
model_methods = st.beta_container()

with ML_model:
    def ml_model():
        url = 'https://raw.githubusercontent.com/yxmauw/General_Assembly_Pub/main/project_2/cloud_app/streamlit_data.csv'
        df = pd.read_csv(url, header=0) # load data
        X = df.drop('SalePrice', axis=1)
        y = df['SalePrice']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        enet_ratio = [.5,.8,.9,.95]
        alpha_l = [1.,10.,100.,500.,1000.] 

        pipe_enet = Pipeline([
                    ('ss', StandardScaler()),
                    ('enet', ElasticNet())
                    ])

        pipe_enet_params = {'enet__alpha': alpha_l,
                        'enet__l1_ratio': enet_ratio
                        }
        cv_ct = 5
        score = 'neg_mean_absolute_error'

        pipe_enet_gs = GridSearchCV(pipe_enet,
                                    pipe_enet_params,
                                    cv=cv_ct,
                                    scoring=score,
                                    verbose=1
                                    )

        pipe_enet_gs.fit(X_train,y_train)

        pickle.dump(pipe_enet_gs, open('final_model.sav','wb'))


with model_methods:
    def predict(new_data):
        # impute missing `Overall Qual` values
        url = 'https://raw.githubusercontent.com/yxmauw/General_Assembly_Pub/main/project_2/cloud_app/streamlit_imp_data.csv'
        imp_data = pd.read_csv(url, header=0)
        imp = KNNImputer()
        imp.fit(imp_data)
        shaped_data = np.reshape(new_data, (1, -1))
        input_data = imp.transform(shaped_data)
        # load model
        with open('project_2/cloud_app/final_model.sav','rb') as f:
            model = pickle.load(f)
        pred = model.predict([input_data][0])
        return pred 
    
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
