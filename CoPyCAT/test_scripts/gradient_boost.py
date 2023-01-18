import os
import glob
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as mse, r2_score


feature_sets = []

# read in features
for df in glob.glob('*.csv'):
    data = pd.read_csv(df)
    data=data.rename(columns={data.columns[0]: "filename"})
    feature_sets.append(data)
# merge feature sets together
features = reduce(lambda left, right: pd.merge(left,right, on='filename'), feature_sets)

# comb_lr_coef_no_sp_renorm

# read in results
results = pd.read_csv('')
# define success metric
outcome_metric = 'comb_lr_coef_no_sp_renorm'
results = results[['filename',outcome_metric]]

# add results to main feature vector
features = pd.merge(features, results, on='filename')
# set filename as series
filename_list = pd.Series(features['filename'])
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
y = np.array(features[outcome_metric])
X = features.drop(outcome_metric, axis=1)

# build regressor
regressor = GradientBoostingRegressor(random_state=1, 
                                      n_estimators=40, 
                                      loss='ls',
                                      learning_rate=0.1)

# instantiate cross validation
cv = KFold(n_splits=10, random_state=123, shuffle=True)

rmse_scores = []
r2_scores = []

# save predictions
preds_main = pd.DataFrame(columns=['filename',outcome_metric+'_actual', outcome_metric+'_predicted'])

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
    preds = pd.DataFrame({'filename': y_filenames, outcome_metric+'_actual': y_test, outcome_metric+'_predicted': y_preds})
    # add predictions to main list
    preds_main=preds_main.append(preds)
    
    # get fold RMSE
    fold_rmse = np.sqrt(mse(y_test, y_preds))
    rmse_scores.append(fold_rmse)
    
    # get R2
    fold_r2 = r2_score(y_test, y_preds)
    r2_scores.append(fold_r2)

# print RMSE scores
print("RMSE score: ", np.mean(rmse_scores))
print("RMSE stndard dev: ", np.std(rmse_scores))
# get average RMSE
print('R2 score: ', np.mean(r2_scores))