import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Funzione per caricare il dataset, applicare SMOTE e addestrare il modello XGBoost
def generate_classification_report(csv_filename, report_filename):
    # Carica il dataset
    data = pd.read_csv(csv_filename)
    
    # Separa le componenti principali e il target (Outcome)
    X = data.drop(columns=['Outcome'])  # Componenti principali ridotte
    y = data['Outcome']
    
    # Applica SMOTE per bilanciare le classi
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    # Suddividi il dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
    
    # Crea e addestra il modello XGBoost
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Prevedi i risultati per il test set
    y_pred = model.predict(X_test)
    
    # Crea il classification report
    report = classification_report(y_test, y_pred)
    
    # Salva il classification report in un file di testo
    with open(report_filename, 'w') as f:
        f.write(f"Classification Report per il dataset {csv_filename}:\n")
        f.write(report)

# Genera i classification report per i due dataset e salva nei file di testo
generate_classification_report('diabetes_pca_2components.csv', 'classification_report_pca2.txt')
generate_classification_report('diabetes_pca_3components.csv', 'classification_report_pca3.txt')

print("I classification report sono stati salvati.")
