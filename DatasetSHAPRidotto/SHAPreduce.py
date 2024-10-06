import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Carica il dataset
data = pd.read_csv('diabetes(2).csv')

# Separa il target (Outcome)
y = data['Outcome']

# 1. Primo dataset: Solo Glucose e Outcome
X_glucose = data[['Glucose']]

# 2. Secondo dataset: Glucose, Age, BMI e Outcome
X_glucose_age_bmi = data[['Glucose', 'Age', 'BMI']]

# Applica SMOTE senza generare nuovi campioni oltre quelli originali
def apply_smote_and_resample(X, y, target_size=768):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    # Applica SMOTE
    X_res, y_res = smote.fit_resample(X, y)
    
    # Combina le classi per il risampling
    combined_resampled = pd.concat([X_res, pd.DataFrame(y_res, columns=['Outcome'])], axis=1)
    
    # Ora riduciamo il dataset a 768 campioni mantenendo il bilanciamento
    resampled_data = resample(combined_resampled, n_samples=target_size, random_state=42, stratify=combined_resampled['Outcome'])
    
    # Restituisci i dati resampled
    X_resampled = resampled_data.drop(columns=['Outcome'])
    y_resampled = resampled_data['Outcome']
    
    return X_resampled, y_resampled

# Applica SMOTE e riduci il numero di campioni a 768 nel dataset con Glucose
X_resampled_glucose, y_resampled_glucose = apply_smote_and_resample(X_glucose, y)

# Applica SMOTE e riduci il numero di campioni a 768 nel dataset con Glucose, Age, BMI
X_resampled_glucose_age_bmi, y_resampled_glucose_age_bmi = apply_smote_and_resample(X_glucose_age_bmi, y)

# Suddividi i dati in train/test set per entrambi i dataset
X_train_glucose, X_test_glucose, y_train_glucose, y_test_glucose = train_test_split(X_resampled_glucose, y_resampled_glucose, test_size=0.2, random_state=42)
X_train_glucose_age_bmi, X_test_glucose_age_bmi, y_train_glucose_age_bmi, y_test_glucose_age_bmi = train_test_split(X_resampled_glucose_age_bmi, y_resampled_glucose_age_bmi, test_size=0.2, random_state=42)

# Crea e addestra il modello Gradient Boosting
model = GradientBoostingClassifier(random_state=42)

# Addestramento su 'Glucose'
model.fit(X_train_glucose, y_train_glucose)
y_pred_glucose = model.predict(X_test_glucose)

# Addestramento su 'Glucose', 'Age' e 'BMI'
model.fit(X_train_glucose_age_bmi, y_train_glucose_age_bmi)
y_pred_glucose_age_bmi = model.predict(X_test_glucose_age_bmi)

# Salva i due dataset finali
# Primo CSV: Solo Glucose e Outcome
glucose_df = pd.DataFrame({'Glucose': X_resampled_glucose['Glucose'], 'Outcome': y_resampled_glucose})
glucose_df.to_csv('glucose_target.csv', index=False)

# Secondo CSV: Glucose, Age, BMI e Outcome
glucose_age_bmi_df = pd.DataFrame({'Glucose': X_resampled_glucose_age_bmi['Glucose'],
                                   'Age': X_resampled_glucose_age_bmi['Age'],
                                   'BMI': X_resampled_glucose_age_bmi['BMI'],
                                   'Outcome': y_resampled_glucose_age_bmi})
glucose_age_bmi_df.to_csv('glucose_age_bmi_target.csv', index=False)

print("I file CSV sono stati salvati correttamente.")
