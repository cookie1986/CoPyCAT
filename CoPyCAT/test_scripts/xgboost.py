import os
import glob
import xgboost as xgb
import numpy as np
import pandas as pd
from functools import reduce
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import mean_squared_error as mse, r2_score


feature_sets = []

# read in features
for df in glob.glob('*.csv'):
    data = pd.read_csv(df)
    data=data.rename(columns={data.columns[0]: "filename"})
    feature_sets.append(data)
# merge feature sets together
featuresDF = reduce(lambda left, right: pd.merge(left,right, on='filename'), feature_sets)

# read in results
resultsDF = pd.read_csv('')

# define success metric
outcome_metrics = ['comb_lr_coef_no_sp_renorm']

for outcome in outcome_metrics:    
    results = resultsDF[['filename',outcome]]
    
    # add results to main feature vector
    features = pd.merge(featuresDF, results, on='filename')
    # set filename as series
    filename_list = pd.Series(featuresDF['filename'])
    # drop filename column
    features = features.drop(features.columns[0], axis=1)
    
    # label encode meta feats
    le = LabelEncoder() 
    features['channel']= le.fit_transform(features['channel'])  
    features['channel_affiliation']= le.fit_transform(features['channel_affiliation']) 
    features['political_affiliation']= le.fit_transform(features['political_affiliation']) 
    
    # one-hot encode meta features
    features = pd.get_dummies(features, columns=['gender_alignment',
                                                  'political_congruence',
                                                  'political_affiliation',
                                                  'channel_affiliation',
                                                  'channel'], drop_first=True)
    
    # create X and Y vars
    y = np.array(features[outcome])
    X = features.drop(outcome, axis=1)
    
    # build regressor
    regressor = XGBRegressor(random_state=1,
                             verbosity=1,
                             sampling_method='uniform',
                             n_estimators=40,
                             max_depth=6,
                             eval_metric='rmse',
                             learning_rate=0.3, # eta
                             n_jobs=-1,
                             reg_lambda=1)
    regressor.get_params
    # instantiate cross validation
    cv = KFold(n_splits=10, random_state=123, shuffle=True)
    
    rmse_scores = []
    r2_scores = []
    
    # save predictions
    preds_main = pd.DataFrame(columns=['filename',outcome+'_actual', outcome+'_predicted'])
    
    # run and evaluate model
    for train_ind, test_ind in cv.split(X, y):
        # get train set
        X_train = X.iloc[train_ind]
        y_train = y[train_ind]
        # get test set
        X_test = X.iloc[test_ind]
        y_test = y[test_ind]
        
        # fit model
        regressor.fit(X_train, y_train)
        
        # generate predictions
        y_preds = regressor.predict(X_test)
        
        # add predictions to main DF
        y_filenames = filename_list[test_ind]
        
        # group test files together
        preds = pd.DataFrame({'filename': y_filenames, outcome+'_actual': y_test, outcome+'_predicted': y_preds})
        # add predictions to main list
        preds_main=preds_main.append(preds)
        
        # get fold RMSE
        fold_rmse = np.sqrt(mse(y_test, y_preds))
        rmse_scores.append(fold_rmse)
        
        # get R2
        fold_r2 = r2_score(y_test, y_preds)
        r2_scores.append(fold_r2)
    
    print(outcome)    
    # print RMSE scores
    print("RMSE score: ", np.mean(rmse_scores))
    print("RMSE stndard dev: ", np.std(rmse_scores))
    # get average RMSE
    print('R2 score: ', np.mean(r2_scores))
    print("")
    print("-------------------------------------")
    print("")