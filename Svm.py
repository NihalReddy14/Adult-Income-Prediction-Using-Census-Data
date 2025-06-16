# Importing necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, log_loss, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Reading Dataset
dataset = pd.read_csv('/content/adult.csv')
dataset.drop(columns=['education'], inplace=True)
dataset.rename(columns={'education.num': 'education_num', 'marital.status': 'marital_status', 'capital.gain': 'capital_gain',
                        'capital.loss': 'capital_loss', 'hours.per.week': 'hours_per_week', 'native.country': 'native_country'}, inplace=True)
dataset['workclass'].replace('?', np.nan, inplace=True)
dataset['occupation'].replace('?', np.nan, inplace=True)
dataset['native_country'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['occupation'], inplace=True)
dataset['native_country'].fillna(dataset['native_country'].mode()[0], inplace=True)
dataset['income'] = dataset['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Label encoding the categorical features
le = LabelEncoder()
cat_list = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
dataset[cat_list] = dataset[cat_list].apply(lambda x: le.fit_transform(x))

dataset.drop(columns=['sex'], inplace=True)

# Slicing dataset into Independent(X) and Target(y) variables
y = dataset.pop('income')
X = dataset

# Scaling the dependent variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Dividing dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Instantiate the SVM classifier
classifier_svm = SVC(probability=True, random_state=0)

# Train the SVM model
model = classifier_svm.fit(X_train, y_train)

# Predictions
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Model evaluation
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Displaying results
print('Confusion Matrix:\n', cm)
print('\nClassification Report:\n', classification_report(y_test, y_pred))
print(f'Roc Auc: {roc_auc:.3f}')
print(f'Accuracy: {accuracy:.3f}')
print(f'F1 Score: {f1score:.3f}')
print(f'Log Loss: {logloss:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')

# Roc Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'SVM, AUC={roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')
plt.show()
