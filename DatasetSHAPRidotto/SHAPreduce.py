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

# 3. Dataset con tutte le feature
X_all = data.drop(columns=['Outcome'])

# Applica SMOTE per bilanciare le classi su entrambi i dataset
smote = SMOTE(random_state=42)

# Applica SMOTE al dataset con solo 'Glucose'
X_smote_glucose, y_smote_glucose = smote.fit_resample(X_glucose, y)

# Applica SMOTE al dataset con 'Glucose', 'Age' e 'BMI'
X_smote_glucose_age_bmi, y_smote_glucose_age_bmi = smote.fit_resample(X_glucose_age_bmi, y)

# Applica SMOTE al dataset con tutte le feature
X_smote_all, y_smote_all = smote.fit_resample(X_all, y)

# Suddividi il dataset in training e test set per 'Glucose'
X_train_glucose, X_test_glucose, y_train_glucose, y_test_glucose = train_test_split(X_smote_glucose, y_smote_glucose, test_size=0.2, random_state=42)

# Suddividi il dataset in training e test set per 'Glucose', 'Age', 'BMI'
X_train_glucose_age_bmi, X_test_glucose_age_bmi, y_train_glucose_age_bmi, y_test_glucose_age_bmi = train_test_split(X_smote_glucose_age_bmi, y_smote_glucose_age_bmi, test_size=0.2, random_state=42)

# Suddividi il dataset in training e test set per tutte le feature
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_smote_all, y_smote_all, test_size=0.2, random_state=42)

# Crea e addestra il modello XGBoost
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

# Addestramento su 'Glucose'
model.fit(X_train_glucose, y_train_glucose)
y_pred_glucose = model.predict(X_test_glucose)

# Addestramento su 'Glucose', 'Age' e 'BMI'
model.fit(X_train_glucose_age_bmi, y_train_glucose_age_bmi)
y_pred_glucose_age_bmi = model.predict(X_test_glucose_age_bmi)

# Addestramento su tutte le feature
model.fit(X_train_all, y_train_all)
y_pred_all = model.predict(X_test_all)

# Crea il classification_report per 'Glucose'
report_glucose = classification_report(y_test_glucose, y_pred_glucose)

# Crea il classification_report per 'Glucose', 'Age' e 'BMI'
report_glucose_age_bmi = classification_report(y_test_glucose_age_bmi, y_pred_glucose_age_bmi)

# Crea il classification_report per tutte le feature
report_all = classification_report(y_test_all, y_pred_all)

# Salva tutto in un file di testo
with open('classification_reports.txt', 'w') as f:
    
    # Scrivi il classification report per Glucose
    f.write("Classification Report con solo Glucose:\n")
    f.write(report_glucose)
    f.write("\n\n")
    
    # Scrivi il classification report per Glucose, Age e BMI
    f.write("Classification Report con Glucose, Age e BMI:\n")
    f.write(report_glucose_age_bmi)
    f.write("\n\n")
    
    # Scrivi il classification report per tutte le feature
    f.write("Classification Report con tutte le feature:\n")
    f.write(report_all)

print("Tutto è stato salvato in 'classification_reports.txt'")
