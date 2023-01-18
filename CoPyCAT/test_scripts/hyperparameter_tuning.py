import os
import glob
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold


feature_sets = []

# read in features
for df in glob.glob('*.csv'):
    data = pd.read_csv(df)
    data=data.rename(columns={data.columns[0]: "filename"})
    feature_sets.append(data)
# merge feature sets together
features = reduce(lambda left, right: pd.merge(left,right, on='filename'), feature_sets)

# read in results
resultsDF = pd.read_csv('')

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
y = np.array(resultsDF['comb_lr_coef_no_sp_renorm'])
X = np.array(features.drop('filename', axis=1))

# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=False, random_state=123)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
	# configure the cross-validation procedure
    cv_inner = KFold(n_splits=3, shuffle=False, random_state=123)
	
    # define the model
    model = RandomForestRegressor(random_state=1, n_jobs=-1, verbose=2)
    
    # build HP space
    space = dict()
    space['n_estimators'] = [40,80,120,160]
    space['max_depth'] = [None, 10,15]
    space['max_features'] = ['auto','sqrt','log2']
    space['min_samples_split'] = [2,3]
    space['min_samples_leaf'] = [1,2,3]
    space['min_impurity_decrease'] = [0.0, 0.1]

	# define search
    search = GridSearchCV(model, space, scoring='neg_mean_squared_error', cv=cv_inner, refit=True)
	# execute search
    result = search.fit(X_train, y_train)
	# get the best performing model fit on the whole training set
    best_model = result.best_estimator_
	
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
	
    # evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
	
    # store the result
    outer_results.append(rmse)
    
	# report progress
    print('>rmse=%.3f, est=%.3f, cfg=%s' % (rmse, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('RMSE: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))