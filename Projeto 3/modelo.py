import pandas as pd
import numpy as np
import catboost as cat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def modelo(dataprep, target):
    
    X_train, X_test,y_train,y_test = train_test_split(dataprep.drop(columns='time_id') , target['target'] , test_size = 0.2, random_state = 42)
    
    scaler=StandardScaler()
    scaler.fit(X_train)
    scaler=StandardScaler()
    scaler.fit(X_train)
    
    las_fit = LassoCV(cv=5)
    las_fit.fit(scaler.transform(X_train), y_train)
   
    
    coef_lin = pd.DataFrame(las_fit.coef_, index=X_train.columns, columns=["lasso_fit"])
    
    tol = 1e-5
    
    final_list_var = list(coef_lin[np.abs(coef_lin['lasso_fit'])>tol].index)
    
    X_train, X_test,y_train,y_test = train_test_split(dataprep[final_list_var],target['target'] , test_size = 0.2, random_state = 42)
    cat_fit = cat.CatBoostRegressor()
    parameters = {'depth'         : [6,7,8,9, 10],
                    'od_type' : ['Iter'],
                    'od_wait' :[1500],      
                    'iterations'    : [18000]
                    }
    grid_cat = GridSearchCV(estimator=cat_fit, param_grid = parameters, cv = 5, n_jobs=-1)
    grid_cat.fit((X_train), y_train, eval_set=((X_test), y_test))


    target["pred_boosting"] = grid_cat.predict(dataprep[final_list_var])
    rmspe = (np.sqrt(np.mean(np.square((target["target"] - target["pred_boosting"]) / target["target"]))))
    return  target, rmspe,grid_cat, final_list_var