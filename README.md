# Adult Income Prediction Using Census Data

This project aims to predict whether an individual's income exceeds $50K per year using the UCI Adult Census dataset. Various classification models like LightGBM, XGBoost, CatBoost, and SVM were applied and tuned to maximize performance.

##  Dataset

- Source: https://www.kaggle.com/datasets/uciml/adult-census-income
- File: `adult.csv`
- Features include: age, workclass, education, marital-status, occupation, etc.

##  Models Implemented

- LightGBM (with and without hyperparameter tuning, PCA, SVD)
- XGBoost
- CatBoost
-  VM (Support Vector Machine)

##  Evaluation Metrics

- Accuracy
- ROC-AUC Score
- Precision & Recall
- GridSearchCV for hyperparameter optimization

##  Project Structure

```bash
.
├── CatBoost.py
├── LightBGM.py
├── LightBGM using gridsearchcv AUC.py
├── Light BGM with pca.py
├── Light BGM with svd.py
├── Ligth BGM with hyperparameter tunes.py
├── Svm.py
├── Xgboost.py
├── adult.csv


git clone https://github.com/NihalReddy14/Adult-Income-Prediction-Using-Census-Data.git
cd Adult-Income-Prediction-Using-Census-Data

