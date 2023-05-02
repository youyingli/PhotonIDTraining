import uproot
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost
import matplotlib.pyplot as plt

import optuna

isEB = False

var = [
    "f_rho"                         ,
    "f_SCRawE"                      ,
    "f_SCEta"                       ,
    "f_etaWidth"                    ,
    "f_phiWidth"                    ,
    "f_sieie"                       ,
    "f_sieip"                       ,
    "f_s4"                          ,
    "f_r9"                          ,
    "f_weight"                      ,

]

if not isEB:
    var.insert(2, "f_ESE")
    var.insert(6, "f_esEffSigmaRR")

with uproot.open('/wk_cms3/youying/public/photon.root') as f:
    tree = f['photoID'].arrays(library='pd')

outdir = 'example'
result_tag = 'EB' if isEB else 'EE'

signal     = tree.loc[ (tree['f_isPrompt'] == True)  & (tree['f_isEB'] == isEB) & (tree['f_pt'] > 18.) ]
background = tree.loc[ (tree['f_isPrompt'] == False) & (tree['f_isEB'] == isEB) & (tree['f_pt'] > 18.) ]

#------------------------------------------------------------------
# Training and test sample preparation
#------------------------------------------------------------------

signal     = signal[var].to_numpy()
background = background[var].to_numpy()

X_sig = signal[:]
X_bkg = background[:]
y_sig = np.ones(len(X_sig))
y_bkg = np.zeros(len(X_bkg))

X_sig_train, X_sig_test,  y_sig_train, y_sig_test  = train_test_split( X_sig, y_sig, test_size = 0.7, random_state = 27 )
X_bkg_train, X_bkg_test,  y_bkg_train, y_bkg_test  = train_test_split( X_bkg, y_bkg, test_size = 0.7, random_state = 27 )

X_sig_train_weight = X_sig_train[:,-1] / np.sum(X_sig_train[:,-1]) * np.sum(X_bkg_train[:,-1])
X_sig_test_weight  = X_sig_test [:,-1] / np.sum(X_sig_test [:,-1]) * np.sum(X_bkg_test [:,-1])
X_bkg_train_weight = X_bkg_train[:,-1]
X_bkg_test_weight  = X_bkg_test [:,-1]

X_train_weight = np.concatenate((X_sig_train_weight, X_bkg_train_weight), axis=0)
X_test_weight  = np.concatenate((X_sig_test_weight, X_bkg_test_weight), axis=0)


X_sig_train = X_sig_train[:,:-1]
X_sig_test  = X_sig_test [:,:-1]
X_bkg_train = X_bkg_train[:,:-1]
X_bkg_test  = X_bkg_test [:,:-1]

X_train = np.concatenate((X_sig_train, X_bkg_train), axis=0)
X_test  = np.concatenate((X_sig_test,  X_bkg_test), axis=0)
y_train = np.concatenate((y_sig_train, y_bkg_train), axis=0)
y_test  = np.concatenate((y_sig_test,  y_bkg_test), axis=0)


#------------------------------------------------------------
# XGBoost training engine
#------------------------------------------------------------

def objective(trial):

    # XGBoost sklearn configuration
    XGBEngine = xgboost.XGBClassifier(
                n_estimators     = 2000,  # Please fix 2000
                learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.1, step=0.005),
                gamma            = trial.suggest_float('gamma', 0, 1, step=0.1),
                max_depth        = trial.suggest_int('max_depth', 3, 19),
                min_child_weight = trial.suggest_float('min_child_weight', 0, 100, step=0.1),
                subsample        = trial.suggest_float('subsample', 0.5, 1, step=0.01),
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1, step=0.01),
                reg_alpha        = trial.suggest_int('reg_alpha', 0, 180),
                reg_lambda       = trial.suggest_float('reg_lambda', 0, 40, step=0.1),
                gpu_id           = 0,
                tree_method      = 'gpu_hist',
                predictor        = 'gpu_predictor',
                eval_metric      = ["logloss"],
                early_stopping_rounds = 10
                )

    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Training
    XGBEngine.fit( X_train, y_train,
                   sample_weight          = X_train_weight,
                   sample_weight_eval_set = [ X_train_weight, X_test_weight ],
                   eval_set               = eval_set,
                   verbose                = False
                   )

    #fpr_test, tpr_test, threshold = metrics.roc_curve(y_test, XGBEngine.predict_proba( X_test )[:,1], pos_label=1, sample_weight=X_test_weight)
    #roc_auc_test = metrics.auc(fpr_test, tpr_test)


    accuracy = metrics.accuracy_score(y_test, XGBEngine.predict( X_test ), sample_weight=X_test_weight)
    print (f" Accuracy = {accuracy}")

    #return roc_auc_test
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=150)

print( f'Best iteration = {study.best_trial.number}' )
print( f'Best AUC = {study.best_trial.value}' )
print( study.best_trial.params )

from optuna.visualization import plot_optimization_history, plot_param_importances

fig = plot_optimization_history(study)
fig.write_image(file=f'{outdir}/optimization_history_{result_tag}.png', format='png')

fig = plot_param_importances(study)
fig.write_image(file=f'{outdir}/param_importances_{result_tag}.png', format='png')
