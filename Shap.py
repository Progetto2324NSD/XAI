import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import numpy as np  # Importa NumPy

# Carica il dataset
data = pd.read_csv('diabetes(2).csv')

# Separa le features e il target
X = data.drop(columns='Outcome')
y = data['Outcome']

# Applica SMOTE per bilanciare le classi
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Suddividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Crea e addestra il modello XGBoost
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Genera le previsioni sul set di test
y_pred = model.predict(X_test)

# Crea il classification_report
report = classification_report(y_test, y_pred)

# Mostra il classification_report a schermo
print("Classification Report con SMOTE e XGBoost:")
print(report)

# Calcola i valori SHAP per il test set
explainer = shap.Explainer(model)  # Usa Explainer per XGBoost
shap_values_test = explainer(X_test)

# Summary plot per visualizzare tutte le feature
plt.figure()
shap.summary_plot(shap_values_test, X_test)

# Visualizza il grafico a barre dell'importanza delle caratteristiche
plt.figure()
shap.plots.bar(shap_values_test)  # Rimuovi 'feature_names'
plt.show()

# Variabile importance plot usando i valori SHAP
importance = np.abs(shap_values_test.values).mean(axis=0)  # Calcola l'importanza delle variabili
importance_df = pd.DataFrame(list(zip(X.columns, importance)), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot dell'importanza delle variabili
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('SHAP Importance')
plt.title('Variable Importance Plot using SHAP Values')
plt.gca().invert_yaxis()  # Inverti l'asse y per mostrare la feature più importante in alto
plt.savefig('variable_importance_plot.png', dpi=300)  # Salva il grafico
plt.show()

# Visualizzazione grafici di dipendenza per le feature più significative
shap.dependence_plot('Glucose', shap_values_test.values, X_test, interaction_index='Glucose')
shap.dependence_plot('Age', shap_values_test.values, X_test, interaction_index='Age')
shap.dependence_plot('BMI', shap_values_test.values, X_test, interaction_index='BMI')

# Decision Plot per il primo campione del test set
shap_values_single = explainer(X_test.iloc[0:1])
shap.decision_plot(explainer.expected_value, shap_values_single.values, X_test.iloc[0:1])

# Decision Plot per 4 campioni
for i in range(4):
    shap_values_single = explainer(X_test.iloc[i:i+1])
    plt.figure()  # Crea una nuova figura per ogni decision plot
    shap.decision_plot(explainer.expected_value, shap_values_single.values, X_test.iloc[i:i+1])
    plt.title(f'Decision Plot for Sample {i + 1}')
    plt.savefig(f'shap_decision_plot_sample_{i + 1}.png', dpi=300)  # Salva il grafico
    plt.show()  # Mostra il grafico
