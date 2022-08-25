# https://www.analyticsvidhya.com/blog/2021/07/streamlit-quickly-turn-your-ml-models-into-web-apps/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.impute import KNNImputer

import streamlit as st

# configuration of the page
st.set_page_config(
    layout="centered", 
    page_icon="🏠", 
    page_title="Are you planning to sell your house?", 
    initial_sidebar_state='auto',
)

def main():
    st.title("🏠Ames Housing Sale Price recommendation tool")
    st.markdown('''
    The algorithm driving this app is built on
    historical housing sale price data to generate
    recommended Sale Price! Please enter your house details 
    to get a Sale Price suggestion 🙂
    ''')
    st.info('Only Enter Numeric Values in the Following Fields')
    
    gr_liv_area = st.number_input('Enter house ground living area in square feet. Accept values 334 to 3395 inclusive', min_value=334.0, max_value=3395.0, value=0.0)
    overall_qual = np.nan
    total_bsmt_sf = st.number_input('Enter house total basement area in square feet. Accept values 0 to 3206 inclusive', min_value=0.0, max_value=3206.0, value=0.0)
    garage_area = st.number_input('Enter house garage area in square feet. Accept values 0 to 1356 inclusive', min_value=0.0, max_value=1356.0, value=0.0)
    year_built = st.number_input('Enter the year your house was built. Accept values 1872 to 2010 inclusive', min_value=0.0, max_value=2010.0, value=0.0)
    mas_vnr_area = st.number_input('Enter house masonry veneer area in square feet. Accept values 0 to 1129 inclusive', min_value=0.0, max_value=1129.0, value=0.0)

    if st.button('Recommend Saleprice'):
        if gr_liv_area and overall_qual and total_bsmt_sf and garage_area and year_built and mas_vnr_area:
            with st.sidebar:
                try: 
                    data = list(gr_liv_area,
                                gr_liv_area**2,
                                gr_liv_area**3,
                                overall_qual, 
                                total_bsmt_sf,
                                garage_area,
                                year_built,
                                mas_vnr_area)
                    result = np.format_float_positional((predict(data)[0]), unique=False, precision=0)
                    st.info(f'# Our SalePrice suggestion is ${result}')
                    st.write('with an estimated uncertainty of ± \$11K')
                except:
                    st.warning('''Oops, looks like you missed a spot. 
                    Please complete all fields to get a quote estimate 
                    for property Sale Price 🙏. 
                    \n\n Thank you. 🙂''')
########################################################## 

@st.cache
def ml_model():
    df = pd.read_csv('streamlit_data.csv') # load data
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # tune hyperparameters
    enet_ratio = [.5,.8,.9,.95]
    alpha_l = [1.,10.,100.,500.,1000.] 
    pipe_enet = Pipeline([
                ('ss', StandardScaler()),
                ('enet', ElasticNet())
                ])
    # instantiate pipeline
    pipe_enet_params = {'enet__alpha': alpha_l,
                        'enet__l1_ratio': enet_ratio
                        }
    cv_ct = 5
    score = 'neg_mean_absolute_error'
    # gridsearch
    pipe_enet_gs = GridSearchCV(pipe_enet,
                                pipe_enet_params,
                                cv=cv_ct,
                                scoring=score,
                                verbose=1
                                )
    # fit model
    model = pipe_enet_gs.fit(X_train,y_train)
    return model
def predict(new_data):
    # impute missing `Overall Qual` values
    imp_data = pd.read_csv('streamlit_imp_data.csv')
    imp = KNNImputer()
    imp.fit(imp_data)
    shaped_data = np.reshape(new_data, (1, -1))
    input_data = imp.transform(shaped_data)
    pred = ml_model().predict([input_data][0])
    return pred 
if __name__=='__main__':
    main()