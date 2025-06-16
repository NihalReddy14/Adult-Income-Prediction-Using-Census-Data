import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, log_loss, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Reading Dataset
dataset = pd.read_csv('/content/adult.csv')
dataset.drop(columns=['education'], inplace=True)
dataset.rename(columns={'education.num':'education_num', 'marital.status':'marital_status', 'capital.gain':'capital_gain',
                          'capital.loss':'capital_loss','hours.per.week':'hours_per_week','native.country':'native_country'}, inplace=True)
dataset['workclass'].replace('?', np.nan, inplace=True)
dataset['occupation'].replace('?', np.nan, inplace=True)
dataset['native_country'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['occupation'], inplace=True)
dataset['native_country'].fillna(dataset['native_country'].mode()[0], inplace=True)

# Label encoding the categorical features
le = LabelEncoder()
cat_list = ['income', 'workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
dataset[cat_list] = dataset[cat_list].apply(lambda x: le.fit_transform(x))
dataset.drop(columns=['sex'], inplace=True)

# Slicing dataset into Independent(X) and Target(y) variables
y = dataset.pop('income')
X = dataset

# Scaling the dependent variables
sc = StandardScaler()
X = sc.fit_transform(X)

# Dividing dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Recursive Feature Elimination with Cross Validation
clf_rf = XGBClassifier(random_state=0)
rfecv = RFECV(estimator=clf_rf, step=1, cv=5, scoring='neg_log_loss')
rfecv = rfecv.fit(X_train, y_train)

# Optimal number of features
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
print('Optimal number of features:', rfecv.n_features_)
print('Best features:', X_train.columns[rfecv.support_])

# Feature Ranking
clf_rf = clf_rf.fit(X_train, y_train)
importances = clf_rf.feature_importances_

# Selecting the Important Features
X_train = X_train.iloc[:, X_train.columns[rfecv.support_]]
X_test = X_test.iloc[:, X_test.columns[rfecv.support_]]

# Building XGBoost Model
classifier_xgb = XGBClassifier(random_state=0)
model_xgb = classifier_xgb.fit(X_train, y_train)

# Predictions and Evaluation for XGBoost
y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = model_xgb.predict(X_test)
print('XGBoost Results:')
print('Confusion Matrix', '\n', confusion_matrix(y_test, y_pred_xgb))
print('\n', 'Classification Report', '\n', classification_report(y_test, y_pred_xgb))

# ROC Curve for XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

# Building CatBoost Model
classifier_cgb = CatBoostClassifier(random_state=0)
model_cgb = classifier_cgb.fit(X_train, y_train)

# Predictions and Evaluation for CatBoost
y_proba_cgb = model_cgb.predict_proba(X_test)[:, 1]
y_pred_cgb = model_cgb.predict(X_test)
print('\nCatBoost Results:')
print('Confusion Matrix', '\n', confusion_matrix(y_test, y_pred_cgb))
print('\n', 'Classification Report', '\n', classification_report(y_test, y_pred_cgb))

# ROC Curve for CatBoost
fpr_cgb, tpr_cgb, _ = roc_curve(y_test, y_proba_cgb)
auc_cgb = roc_auc_score(y_test, y_proba_cgb)

# Plot ROC Curve for both XGBoost and CatBoost
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost, AUC={:.3f}".format(auc_xgb))
plt.plot(fpr_cgb, tpr_cgb, label="CatBoost, AUC={:.3f}".format(auc_cgb))
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')
plt.show()
