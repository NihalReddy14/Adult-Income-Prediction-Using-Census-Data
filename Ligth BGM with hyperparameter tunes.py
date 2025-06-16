import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, log_loss, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

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
clf_rf = lgb.LGBMClassifier(random_state=0)
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

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'num_leaves': [31, 50, 100],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'class_weight': ['balanced', None],  # Handle imbalanced classes
}

grid_search = GridSearchCV(estimator=lgb.LGBMClassifier(random_state=0), param_grid=param_grid, cv=5, scoring='neg_log_loss')
grid_search.fit(X_train, y_train)

# Best parameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Building LightGBM Model with Best Hyperparameters
classifier_lgb = lgb.LGBMClassifier(random_state=0, **best_params)
model_lgb = classifier_lgb.fit(X_train, y_train)

# Predictions and Evaluation for LightGBM
y_proba_lgb = model_lgb.predict_proba(X_test)[:, 1]
y_pred_lgb = model_lgb.predict(X_test)
print('LightGBM Results:')
print('Confusion Matrix', '\n', confusion_matrix(y_test, y_pred_lgb))
print('\n', 'Classification Report', '\n', classification_report(y_test, y_pred_lgb))

# ROC Curve for LightGBM
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba_lgb)
auc_lgb = roc_auc_score(y_test, y_proba_lgb)

# Plot ROC Curve for LightGBM
plt.figure(figsize=(8, 6))
plt.plot(fpr_lgb, tpr_lgb, label="LightGBM, AUC={:.3f}".format(auc_lgb))
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')
plt.show()
