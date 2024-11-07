import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek

import warnings
warnings.filterwarnings('ignore')

# Шаг 1. Создайте несбалансированный набор данных двоичной классификации.
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8,
                           weights=[0.9, 0.1], flip_y=0, random_state=42)
np.unique(y, return_counts=True)

# Разделите набор данных на обучающий и тестовый наборы.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

### Эксперимент 1: Обучите классификатор логистической регрессии
log_reg = LogisticRegression(C=1, solver='liblinear')
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
# print(classification_report(y_test, y_pred_log_reg))

### Эксперимент 2: Обучение  Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=30, max_depth=3)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
# print(classification_report(y_test, y_pred_rf))

### Эксперимент 3: Обучение XGBoost
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
# print(classification_report(y_test, y_pred_xgb))

### Эксперимент 4. Устраните дисбаланс классов с помощью SMOTETomek, а затем обучите XGBoost.
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
np.unique(y_train_res, return_counts=True)

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_clf.predict(X_test)
# print(classification_report(y_test, y_pred_xgb))

models = [
    (
        "Logistic Regression",
        LogisticRegression(C=1, solver='liblinear'),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=30, max_depth=3),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "XGBClassifier",
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "XGBClassifier With SMOTE",
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        (X_train_res, y_train_res),
        (X_test, y_test)
    )
]

reports = []

for model_name, model, train_set, test_set in models:
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    reports.append(report)

import mlflow
import mlflow.sklearn
import mlflow.xgboost


# Initialize MLflow

experiment_name = "Anomaly Detection"
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Проверяем, существует ли эксперимент, и создаем его, если нет
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)


import seaborn as sns
# Создайте DataFrame для хранения метрик.
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Recall_Class_1', 'Recall_Class_0', 'F1_Score_Macro'])

for i, element in enumerate(models):
    model_name = element[0]
    model = element[1]
    report = reports[i]

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_metric('accuracy', report['accuracy'])
        mlflow.log_metric('recall_class_1', report['1']['recall'])
        mlflow.log_metric('recall_class_0', report['0']['recall'])
        mlflow.log_metric('f1_score_macro', report['macro avg']['f1-score'])

        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Добавьте метрики в DataFrame.
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [report['accuracy']],
            'Recall_Class_1': [report['1']['recall']],
            'Recall_Class_0': [report['0']['recall']],
            'F1_Score_Macro': [report['macro avg']['f1-score']]
        })], ignore_index=True)


# Создайте гистограмму показателей
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='value', hue='variable', data=pd.melt(metrics_df, ['Model']))
plt.title('Comparison of Metrics for Different Models')
plt.ylabel('Score')
plt.xlabel('Model')
plt.savefig('metrics_plot.png')

# Сохраните график как файл изображения.
plt.savefig('metrics_plot.png')

# Зарегистрируйте график как артефакт
mlflow.log_artifact('metrics_plot.png')

