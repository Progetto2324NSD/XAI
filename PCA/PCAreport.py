# Importa le librerie necessarie
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pandas as pd

# Definisci una funzione per caricare i dati, addestrare il modello e stampare il classification report
def train_and_report(file_path, components):
    # Carica il dataset
    data = pd.read_csv(file_path)

    # Prepara le feature e il target
    X = data.iloc[:, :-1]
    y = data['Outcome']
    # Dividi i dati in set di addestramento e set di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inizializza e allena l'XGBClassifier con learning_rate=0.05
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                          learning_rate=0.05, n_estimators=200, max_depth=4)
    model.fit(X_train, y_train)

    # Predici il set di test
    y_pred = model.predict(X_test)

    # Genera il Classification Report
    report = classification_report(y_test, y_pred)
    print(f"\nClassification Report per {components} Componenti:\n")
    print(report)

# Percorsi ai file CSV per 2 e 3 componenti
file_path_2_components = 'diabetes_pca_2components.csv'  # Sostituisci con il percorso effettivo del file
file_path_3_components = 'diabetes_pca_3components.csv'  # Sostituisci con il percorso effettivo del file

# Testa e stampa il Classification Report per 2 componenti
train_and_report(file_path_2_components, 2)

# Testa e stampa il Classification Report per 3 componenti
train_and_report(file_path_3_components, 3)
