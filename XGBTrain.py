import uproot
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost
import matplotlib.pyplot as plt
import xgboost2tmva

isEB = True

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
# Draw correlation matrix
#------------------------------------------------------------------
#var_modified = var[:-1]
#sig_correlations =  signal[var_modified].corr()
#bkg_correlations =  background[var_modified].corr()
#
#x = plt.subplots(figsize=(10,10))
#sns.heatmap(sig_correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
#            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
#                                            )
#plt.savefig('sig_correlations.png')
#plt.savefig('sig_correlations.pdf')
#
#
#x = plt.subplots(figsize=(10,10))
#sns.heatmap(bkg_correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
#            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
#                                            )
#plt.savefig('bkg_correlations.png')
#plt.savefig('bkg_correlations.pdf')


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

# Need to apply pt-eta weight to signal samples and background sample is fixed to 1
# Sum of sample weights should be modified to be the same as one in background to avoid unbalanced issue
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

# XGBoost sklearn configuration
XGBEngine = xgboost.XGBClassifier(
            n_estimators     =  2000,  # Please fix 2000
            learning_rate    =  0.1,  # Please fix 0.01
            max_depth        =  6,
            #min_child_weight =  5,
            #gamma            =  0,
            #subsample        =  0.7,
            #colsample_bytree =  0.9,
            #reg_alpha        =  1,
            #reg_lambda       =  0,
            #objective        = 'binary:logitraw',
            #scale_pos_weight =  float( len(y_train_b) ) / len(y_train_s)
            )


eval_set = [(X_train, y_train), (X_test, y_test)]

# Training
XGBEngine.fit( X_train, y_train,
               sample_weight          = X_train_weight,
               sample_weight_eval_set = [ X_train_weight, X_test_weight ],
               eval_metric            = ["logloss"],
               eval_set               = eval_set,
               early_stopping_rounds  = 10,
               verbose                = True
               )


#------------------------------------------------------------
# Loss function
#------------------------------------------------------------

results = XGBEngine.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Valid')
ax.legend()
ax.set_ylabel('Log Loss')
ax.set_yscale('log')
ax.set_title('XGBoost Log Loss')
plt.savefig(f'{outdir}/logloss.png')
plt.savefig(f'{outdir}/logloss.pdf')


#------------------------------------------------------------
# Training Accuracy
#------------------------------------------------------------

y_sig_train_pred = XGBEngine.predict_proba( X_sig_train )[:,1]
y_bkg_train_pred = XGBEngine.predict_proba( X_bkg_train )[:,1]
y_sig_test_pred  = XGBEngine.predict_proba( X_sig_test ) [:,1]
y_bkg_test_pred  = XGBEngine.predict_proba( X_bkg_test ) [:,1]

accuracy = metrics.accuracy_score(y_test, XGBEngine.predict( X_test ), sample_weight=X_test_weight)
print (" Accuracy = %f" % accuracy)


#------------------------------------------------------------
# ROC curve
#------------------------------------------------------------

#Train ROC
fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train, np.concatenate((y_sig_train_pred, y_bkg_train_pred)), pos_label=1, sample_weight=X_train_weight)
roc_auc_train = metrics.auc(fpr_train, tpr_train)
rfpr_train = [ ( 1. - i ) for i in fpr_train ]

#Test ROC
fpr_test, tpr_test, threshold = metrics.roc_curve(y_test, np.concatenate((y_sig_test_pred, y_bkg_test_pred)), pos_label=1, sample_weight=X_test_weight)
roc_auc_test = metrics.auc(fpr_test, tpr_test)
rfpr_test = [ ( 1. - i ) for i in fpr_test ]


fig, ax = plt.subplots()
ax.set_title('Receiver Operating Characteristic')
ax.plot(tpr_train, np.array(rfpr_train), 'b', label = 'AUC (Train) = %0.2f' % roc_auc_train)
ax.plot(tpr_test,  np.array(rfpr_test),  'g', label = 'AUC (Test)  = %0.2f' % roc_auc_test)
ax.legend(loc = 'lower right')
ax.plot([0, 1], [1, 0],'--')
ax.set_ylabel('Signal efficiency')
ax.set_xlabel('Background reduction')
plt.savefig(f'{outdir}/roc.png')
plt.savefig(f'{outdir}/roc.pdf')


#------------------------------------------------------------
# Training and Test histogram comparison
#------------------------------------------------------------
fig, ax = plt.subplots()

ax.hist( y_sig_test_pred , bins=60, label='sig (Test)', weights=X_sig_test_weight, histtype='step' )
ax.hist( y_bkg_test_pred , bins=60, label='big (Test)', weights=X_bkg_test_weight, histtype='step' )

ax.set_yscale('log')
ax.legend(loc='upper right')
plt.savefig(f'{outdir}/hist.png')
plt.savefig(f'{outdir}/hist.pdf')


#------------------------------------------------------------
# Feature importance (how many times )
#------------------------------------------------------------
axes = xgboost.plot_importance(XGBEngine)
plt.savefig(f'{outdir}/importance.png')
plt.savefig(f'{outdir}/importance.pdf')


#------------------------------------------------------------
# Feature importance (how many times )
#------------------------------------------------------------
xgboost2tmva.convert_model( XGBEngine.get_booster().get_dump(),
                            input_variables=[ ('f{i}','F') for i in range(len(var)-1) ],
                            output_xml=f'{outdir}/photonID_xgboost_{result_tag}.xml'
                            )

XGBEngine.save_model(f'{outdir}/photonID_xgboost_{result_tag}.xgb')
XGBEngine.save_model(f'{outdir}/photonID_xgboost_{result_tag}.json')

import xml.etree.ElementTree as ET
tree = ET.parse(f'{outdir}/photonID_xgboost_{result_tag}.xml')
ET.indent(tree)
tree.write(f'{outdir}/photonID_xgboost_{result_tag}.xml')
