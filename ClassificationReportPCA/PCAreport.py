import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Carica il dataset ridotto con PCA
data = pd.read_csv('diabetes_reduced.csv')

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
with open('classification_report_pca.txt', 'w') as f:
    f.write("Classification Report sul dataset ridotto con PCA:\n")
    f.write(report)

print("Il classification report è stato salvato in 'classification_report_pca.txt'")
