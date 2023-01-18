import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error as mse

# read in results
results = pd.read_csv('combined_minmax.csv')
# set filename as series
filename_list = pd.Series(results['filename'])
# define success metric
results = results[['filename','comb_lr_coef_no_sp_renorm']]

# isolate scores
scores = results.iloc[:, 1]

# instantiate cross validation
cv = KFold(n_splits=10, random_state=123, shuffle=True)

# empty lists to store scores
baseline_r2 = []
baseline_rmse = []

# save predictions
preds_main = pd.DataFrame(columns=['filename','baseline_actual', 'baseline_predicted'])

# calculate the mean score in training fold
for train_index, test_index in cv.split(scores):
    train_mean = scores[train_index].mean()
    
    baseline_preds = np.full(shape=len(test_index), fill_value=train_mean)
    
    y_test = scores[test_index]
    
    # add predictions to main DF
    y_filenames = filename_list[test_index]
    
    # group test files together
    preds = pd.DataFrame({'filename': y_filenames, 'baseline_actual': y_test, 'baseline_predicted': baseline_preds})
    # add predictions to main list
    preds_main=preds_main.append(preds)
    
    # generate r2
    r2 = r2_score(baseline_preds, y_test)
    baseline_r2.append(r2)
    
    # get RMSE
    rmse = np.sqrt(mse(baseline_preds, y_test))
    baseline_rmse.append(rmse)

print('Baseline R2: '+str(np.mean(baseline_r2)))
print('Baseline RMSE: '+str(np.mean(baseline_rmse)))
print("RMSE stndard dev: ", np.std(baseline_rmse))