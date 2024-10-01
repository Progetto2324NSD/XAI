import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Carica il dataset
data = pd.read_csv('diabetes(2).csv')

# Separa le features e il target
X = data.drop(columns='Outcome')  # Supponendo che 'Outcome' sia la colonna target
y = data['Outcome']

# Suddividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea e addestra il modello XGBoost
model = xgb.XGBClassifier(eval_metric='logloss')  # rimosso use_label_encoder
model.fit(X_train, y_train)

# Calcola i valori SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_train)

# Stampa le forme di shap_values e X_train per la verifica
print("Forma dei valori SHAP:", shap_values.shape)
print("Shape relativo ad X_train:", X_train.shape)

# Crea il summary plot per visualizzare tutte le feature
shap.summary_plot(shap_values, X_train, max_display=X_train.shape[1])

# Visualizza il grafico di dipendenza per una caratteristica specifica
shap.dependence_plot("Glucose", shap_values.values, X_train)

# Mostra il grafico di forza per il primo campione del set di test
shap.initjs()  # Inizializza JavaScript per il grafico interattivo

# Assicurati di utilizzare l'oggetto Explanation per il campione specifico
shap_values_single = explainer(X_test.iloc[0:1])  # Calcola i valori SHAP per il primo campione
shap.force_plot(explainer.expected_value, shap_values_single.values, X_test.iloc[0:1])

# Visualizza il grafico a barre dell'importanza delle caratteristiche
shap.plots.bar(shap_values)

# Mostra il grafico decisionale per il primo campione
shap.decision_plot(explainer.expected_value, shap_values_single.values, X_test.iloc[0:1])
