import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.impute import KNNImputer
import pickle

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