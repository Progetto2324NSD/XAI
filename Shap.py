import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Carica il dataset
data = pd.read_csv('diabetes(2).csv')

# Separa le features e il target
X = data.drop(columns='Outcome') 
y = data['Outcome']

# Suddividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea e addestra il modello XGBoost
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Genera le previsioni sul set di test
y_pred = model.predict(X_test)

# Crea il classification_report
report = classification_report(y_test, y_pred)

# Salva il classification_report in un file di testo
with open('classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

# Mostra il classification_report a schermo (opzionale)
print("Classification Report:")
print(report)

# Calcola i valori SHAP per il test set con l'oggetto explainer
explainer = shap.TreeExplainer(model)
shap_values_test = explainer(X_test)  

# Stampa le forme di shap_values e X_test per la verifica
print("Forma dei valori SHAP (test set):", shap_values_test.shape)
print("Shape relativo ad X_test:", X_test.shape)

# Summary plot per visualizzare tutte le feature
plt.figure()
shap.summary_plot(shap_values_test, X_test, max_display=X_test.shape[1])
plt.savefig('shap_summary_plot.eps', format='eps', dpi=1200)

# Visualizzazione grafici di dipendenza (Riportate le voci più significative Glucose, Age e BMI)
plt.figure() 
shap.dependence_plot('Glucose', shap_values_test.values, X_test, interaction_index='Glucose', show=False)
plt.show()

shap.dependence_plot('Age', shap_values_test.values, X_test, interaction_index='Age', show=False)
plt.show()

shap.dependence_plot('BMI', shap_values_test.values, X_test, interaction_index='BMI', show=False)
plt.show()

# Mostra il grafico di forza per il primo campione del set di test
shap.initjs()

# Grafico di forza per il singolo campione di set Force Plot
shap_values_single = explainer(X_test.iloc[0:1])
shap.force_plot(explainer.expected_value, shap_values_single.values, X_test.iloc[0:1])

# Visualizza il grafico a barre dell'importanza delle caratteristiche
plt.figure()
shap.plots.bar(shap_values_test)
plt.savefig('shap_bar_plot.eps', format='eps', dpi=1200) 

# Mostra il grafico decisionale per il primo campione
plt.figure()  # Crea una nuova figura per il plot
shap.decision_plot(explainer.expected_value, shap_values_single.values, X_test.iloc[0:1])
plt.savefig('shap_decision_plot.eps', format='eps', dpi=1200)

#Decision Plot per + Campioni
for i in range(4):  
    shap.decision_plot(explainer.expected_value, shap_values_test.values[i], X_test.iloc[i:i+1])
    plt.savefig(f'shap_decision_plot_sample_{i + 1}.eps', format='eps', dpi=1200)  
    plt.show()
