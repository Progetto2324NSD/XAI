import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Carica il dataset
data = pd.read_csv('diabetes(2).csv')

# Separa il target (Outcome)
y = data['Outcome']

# 1. Dataset con solo la feature 'Glucose'
X_glucose = data[['Glucose']]

# 2. Dataset con le feature 'Glucose', 'Age' e 'BMI'
X_glucose_age_bmi = data[['Glucose', 'Age', 'BMI']]

# Applica SMOTE per bilanciare le classi su entrambi i dataset
smote = SMOTE(random_state=42)

# Applica SMOTE al dataset con solo 'Glucose'
X_smote_glucose, y_smote_glucose = smote.fit_resample(X_glucose, y)

# Applica SMOTE al dataset con 'Glucose', 'Age' e 'BMI'
X_smote_glucose_age_bmi, y_smote_glucose_age_bmi = smote.fit_resample(X_glucose_age_bmi, y)

# Suddividi il dataset in training e test set per 'Glucose'
X_train_glucose, X_test_glucose, y_train_glucose, y_test_glucose = train_test_split(X_smote_glucose, y_smote_glucose, test_size=0.2, random_state=42)

# Suddividi il dataset in training e test set per 'Glucose', 'Age', 'BMI'
X_train_glucose_age_bmi, X_test_glucose_age_bmi, y_train_glucose_age_bmi, y_test_glucose_age_bmi = train_test_split(X_smote_glucose_age_bmi, y_smote_glucose_age_bmi, test_size=0.2, random_state=42)

# Crea e addestra il modello XGBoost
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

# Addestramento su 'Glucose'
model.fit(X_train_glucose, y_train_glucose)
y_pred_glucose = model.predict(X_test_glucose)

# Addestramento su 'Glucose', 'Age' e 'BMI'
model.fit(X_train_glucose_age_bmi, y_train_glucose_age_bmi)
y_pred_glucose_age_bmi = model.predict(X_test_glucose_age_bmi)

# Crea il classification_report per 'Glucose'
report_glucose = classification_report(y_test_glucose, y_pred_glucose)

# Crea il classification_report per 'Glucose', 'Age', 'BMI'
report_glucose_age_bmi = classification_report(y_test_glucose_age_bmi, y_pred_glucose_age_bmi)

# Salva i report in un file di testo
with open('classification_reports.txt', 'w') as f:
    f.write("Classification Report con solo Glucose:\n")
    f.write(report_glucose)
    f.write("\nClassification Report con Glucose, Age e BMI:\n")
    f.write(report_glucose_age_bmi)

print("I classification report sono stati salvati in 'classification_reports.txt'")
