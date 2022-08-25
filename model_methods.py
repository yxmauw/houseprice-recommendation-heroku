import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.impute import KNNImputer
import pickle
    
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
