# Telecom_churn_analysis using several ML algorithms
This repository contains data and notebooks for a telecom churn analysis project. It includes raw datasets, notebooks that perform preprocessing, model training (Random Forest and XGBoost), and hybridization experiments.

Repository structure
--------------------
- `dataset/` — dataset files used for experiments
	- `churn-bigml-80.csv` — main dataset (80%)
	- `churn-bigml-20.csv` — smaller dataset (20%)
	- `kaggle.json`, `telecom-churn-datasets.zip` — support files
- `notebooks/` — Jupyter notebooks
	- `01)download_dataset.ipynb` — notebook to download or prepare datasets
	- `02)data_preprocessing.ipynb` — preprocessing pipeline, SMOTE balancing, Random Forest and XGBoost training and tuning. This is the main notebook that contains the code reproduced by the dashboard code previously developed.
	- `03)Hybridization.ipynb` — hybrid / stacking experiments (Random Forest + ANN etc.) & and catboost,lightgbm,gradientclassifier compariosn & tuning to achieve more accuracy.
	- `catboost_info/` — CatBoost training logs and related files
- `rf_xgbost/` — local virtualenv folder (if present) — may contain a Python environment used during development
- `.gitignore`, other meta files

Discussion
----------------
-accuracy of the prediction with Random Forest before fine tuning:
*Training  scores:
	Accuracy=0.945

*Validating  scores:
	Accuracy=0.921


-here accuracy metricse shows a gap between training and testing data prediction metrics, whic means there is a overfiiting in traing. To avoid overfitting i changes parameters (number of trees in the forest and max_depth) to achieve a better predictions.After training the data set with newly chnaged parameters.:
rf_model_after_tuning=RandomForestClassifier(
    n_estimators=200, #number of trees
    random_state=42, #for reproducibility
    max_depth=7,
    min_samples_split=10,
    min_samples_leaf=3,
    max_features='sqrt'
).

-performanc eevaluation after changing parameters:
Training  scores:
	Accuracy=0.886

Validating  scores:
	Accuracy=0.908

-now model learn the pattern without overfitting.

Here is the performance comparison of XGBOOST & RF
Random Forest Accuracy: 0.9082397003745318
XGBoost Accuracy: 0.9044943820224719

Random Forest Report:
               precision    recall  f1-score   support

           0       0.95      0.94      0.95       455
           1       0.68      0.71      0.70        79

    accuracy                           0.91       534
   macro avg       0.82      0.83      0.82       534
weighted avg       0.91      0.91      0.91       534


XGBoost Report:
               precision    recall  f1-score   support

           0       0.95      0.93      0.94       455
           1       0.66      0.75      0.70        79

    accuracy                           0.90       534
   macro avg       0.81      0.84      0.82       534
weighted avg       0.91      0.90      0.91       534
