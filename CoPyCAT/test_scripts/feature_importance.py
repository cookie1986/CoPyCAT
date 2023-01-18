import os
import glob
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict
from scipy.cluster import hierarchy
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.inspection import permutation_importance

# empty list to store feature sets
feature_sets = []

# read in features
for df in glob.glob('*.csv'):
    data = pd.read_csv(df)
    data=data.rename(columns={data.columns[0]: "filename"})
    feature_sets.append(data)
# merge feature sets together
featuresDF = reduce(lambda left, right: pd.merge(left,right, on='filename'), feature_sets)

# read in results
results = pd.read_csv('')
# define success metric
outcome_metric = 'comb_lr_coef_no_sp_renorm'
results = results[['filename',outcome_metric]]

# add results to main feature vector
featuresDF = pd.merge(featuresDF, results, on='filename')
# drop filename column
featuresDF = featuresDF.drop(featuresDF.columns[0], axis=1)

# label encode meta feats
le = LabelEncoder() 
featuresDF['channel']= le.fit_transform(featuresDF['channel'])  
featuresDF['channel_affiliation']= le.fit_transform(featuresDF['channel_affiliation']) 
featuresDF['political_affiliation']= le.fit_transform(featuresDF['political_affiliation']) 

# one-hot encode meta features
featuresDF = pd.get_dummies(featuresDF, columns=['gender_alignment',
                                              'political_congruence',
                                              'political_affiliation',
                                              'channel_affiliation',
                                              'channel'], drop_first=True)

# get list of features
x_names = list(featuresDF.columns.values.tolist())
x_names = [name for name in x_names if name != outcome_metric]

# create X and Y vars
y = np.array(featuresDF[outcome_metric])
X = featuresDF.drop(outcome_metric, axis=1)

# dendogram of correlations between features
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(
    corr_linkage, labels=x_names, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro['ivl']))
# display
# ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
# ax2.set_xticks(dendro_idx)
# ax2.set_yticks(dendro_idx)
# ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
# ax2.set_yticklabels(dendro['ivl'])
# fig.tight_layout()
# plt.show()

# build feature clusters
cluster_ids = hierarchy.fcluster(corr_linkage, 2.5, criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

# get subset of feature names
selected_feature_names = [x_names[i] for i in selected_features]
# create subset of X features
X = X.loc[:, selected_feature_names]

# instantiate random forest
regressor = ExtraTreesRegressor(random_state=1, 
                                          n_estimators=40, 
                                          n_jobs=-1)

# instantiate cross validation
cv = KFold(n_splits=10, random_state=123, shuffle=True)

# feature importance scores
fiDF = pd.DataFrame(columns=selected_feature_names)

# empty list to store full model RMSE
fullmod_rmse = []

# extract feature importance scores
for train_ind, test_ind in cv.split(X, y):
    X_train = X.iloc[train_ind]
    X_test = X.iloc[test_ind]
    y_train = y[train_ind]
    y_test = y[test_ind]
    # fit random froest
    regressor.fit(X_train, y_train)
    
    # get predictions to establish the baseline performance level
    y_preds = regressor.predict(X_test)
    
    # measure RMSE per fold
    fold_rmse = np.sqrt(mse(y_test, y_preds))
    
    # add fold rmse to main list
    fullmod_rmse.append(fold_rmse)
    
    # calculate permutation importance scores
    r = permutation_importance(regressor, X_test, y_test,
                               n_repeats=5,
                               random_state=123,
                               scoring='neg_mean_squared_error')
    # empty dict to store importance scores
    feature_imp = {}
    # get imp scores and add to dict
    for i in r.importances_mean.argsort()[::-1]:
        feature_imp.update({selected_feature_names[i]: round(r.importances_mean[i],5)})
        
    # add to dataframe
    fiDF=fiDF.append(feature_imp, ignore_index=True)

# print baseline rmse performance (without permutation)
print("Baseline RMSE (without permutation): ", np.mean(fullmod_rmse))

# get mean of each feature
feature_means = fiDF.mean().sort_values(ascending=False)
print(feature_means.head(10))