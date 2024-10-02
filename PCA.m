clc;
clear;

% Carica il dataset
data = readtable('diabetes(2).csv');

% Controlla i dati caricati
disp('Dati originali:');
disp(head(data));  % Mostra le prime righe del dataset

% Separa le features (X) dal target (y)
X = data{:, 1:end-1}; % Assumiamo che l'ultima colonna sia il target
y = data{:, end};     % Target (esito della malattia)

% Mostra dimensioni di X e y
disp(['Dimensione di X: ', num2str(size(X))]);
disp(['Dimensione di y: ', num2str(size(y))]);

% Esegui la PCA sul dataset X (le features)
[coeff, score, latent, tsquared, explained, mu] = pca(X);

% Mostra la varianza spiegata
disp('Varianza spiegata da ogni componente principale:');
disp(explained);

% Visualizza il grafico di Pareto della varianza spiegata
figure;
pareto(explained);
title('Grafico di Pareto della Varianza Spiegata dalle Componenti Principali');
xlabel('Componenti Principali');
ylabel('Varianza Spiegata (%)');

% Mostra anche la varianza cumulativa
hold on; 
cumulativeVar = cumsum(explained); 
yyaxis right;
plot(1:length(cumulativeVar), cumulativeVar, '-o', 'Color', 'r', 'LineWidth', 2);
ylabel('Varianza Cumulativa (%)');
legend('Varianza Spiegata (%)', 'Varianza Cumulativa (%)');
hold off; 
cumulativeExplained = cumsum(explained);

% Determina il numero di componenti che spiegano almeno il 95% della varianza
numComponents95 = find(cumulativeExplained >= 95, 1);
disp(['Numero di componenti usati per spiegare almeno il 95% della varianza: ', num2str(numComponents95)]);

% Calcola i coefficienti delle componenti principali
feature_importance = abs(coeff(:, 1:numComponents95));
importance_scores = sum(feature_importance, 2);

% Ordina le feature in base all'importanza
[~, sorted_indices] = sort(importance_scores, 'descend');
sorted_features = data.Properties.VariableNames(1:end-1);

% Mostra le feature originali più importanti
disp('Feature originali ordinate per importanza:');
disp(sorted_features(sorted_indices));

% Selezione delle feature rilevanti
median_importance = median(importance_scores);
relevant_features_indices = sorted_indices(importance_scores(sorted_indices) > median_importance);
relevant_features = sorted_features(relevant_features_indices);

% Mostra le feature più rilevanti
disp('Feature più rilevanti da mantenere:');
disp(relevant_features');

% Costruisci un nuovo dataset con le feature rilevanti
X_reduced = X(:, relevant_features_indices);
reduced_data = array2table(X_reduced, 'VariableNames', relevant_features);
reduced_data.Outcome = y;

% Mostra il dataset ridotto
disp('Dataset ridotto con le feature più rilevanti:');
disp(head(reduced_data));

% Salva il dataset ridotto in un file CSV
writetable(reduced_data, 'diabetes_reduced.csv');
disp('Dataset ridotto salvato in diabetes_reduced.csv.');

%SCATTER PLOT
% Visualizzazione Scatter Plot 2D - Prime 2 Componenti Principali
figure;
scatter(score(:,1), score(:,2), 50, y, 'filled');
title('Scatter Plot 2D - Prime 2 Componenti Principali');
xlabel('Prima componente principale');
ylabel('Seconda componente principale');
colorbar;
grid on;

% Visualizzazione Scatter Plot 3D - Prime 3 Componenti Principali
figure;
scatter3(score(:,1), score(:,2), score(:,3), 50, y, 'filled');
title('Scatter Plot 3D - Prime 3 Componenti Principali');
xlabel('Prima componente principale');
ylabel('Seconda componente principale');
zlabel('Terza componente principale');
colorbar;
grid on;
