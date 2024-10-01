import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Carica il dataset
data = pd.read_csv('diabetes(2).csv')

# Separa le features e il target
X = data.drop(columns='Outcome')  # Supponendo che 'Outcome' sia la colonna target
y = data['Outcome']

# Suddividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea e addestra il modello XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Crea un oggetto LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['No Diabetes', 'Diabetes'], mode='classification')

# Seleziona un campione dal test set da spiegare
sample = X_test.iloc[0]  # Cambia l'indice per spiegare un campione diverso

# Ottieni le spiegazioni
exp = explainer.explain_instance(sample.values, model.predict_proba)

# Stampa le spiegazioni
print(exp.as_list())

# Visualizza le spiegazioni in un grafico
fig = exp.as_pyplot_figure()
plt.title("Importanza delle funzionalità LIME per una singola previsione")
plt.show()
