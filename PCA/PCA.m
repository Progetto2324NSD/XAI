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

% Calcola la varianza cumulativa
cumulativeExplained = cumsum(explained);

% Determina il numero di componenti che spiegano almeno il 95% della varianza
numComponents95 = find(cumulativeExplained >= 95, 1);
disp(['Numero di componenti usati per spiegare almeno il 95% della varianza: ', num2str(numComponents95)]);

%% Salva il dataset ridotto con 2 componenti principali

% Seleziona solo le prime 2 componenti principali
X_reduced_2 = score(:, 1:2); % Usa solo le prime due colonne di 'score'

% Aggiungi la colonna target al dataset ridotto
X_reduced_2_with_target = array2table(X_reduced_2, 'VariableNames', {'PC1', 'PC2'});
X_reduced_2_with_target.Outcome = y;

% Mostra il dataset ridotto con 2 componenti principali
disp('Dataset ridotto con le prime 2 componenti principali e la colonna target:');
disp(head(X_reduced_2_with_target));

% Salva il dataset ridotto in un file CSV
writetable(X_reduced_2_with_target, 'diabetes_pca_2components.csv');
disp('Dataset ridotto con 2 componenti salvato in diabetes_pca_2components.csv.');

%% Salva il dataset ridotto con 3 componenti principali

% Seleziona le prime 3 componenti principali
X_reduced_3 = score(:, 1:3); % Usa le prime tre colonne di 'score'

% Aggiungi la colonna target al dataset ridotto
X_reduced_3_with_target = array2table(X_reduced_3, 'VariableNames', {'PC1', 'PC2', 'PC3'});
X_reduced_3_with_target.Outcome = y;

% Mostra il dataset ridotto con 3 componenti principali
disp('Dataset ridotto con le prime 3 componenti principali e la colonna target:');
disp(head(X_reduced_3_with_target));

% Salva il dataset ridotto in un file CSV
writetable(X_reduced_3_with_target, 'diabetes_pca_3components.csv');
disp('Dataset ridotto con 3 componenti salvato in diabetes_pca_3components.csv.');

%% SCATTER PLOT
% Visualizzazione Scatter Plot 2D - Prime 2 Componenti Principali
figure;
scatter(X_reduced_2(:,1), X_reduced_2(:,2), 50, y, 'filled');
title('Scatter Plot 2D - Prime 2 Componenti Principali');
xlabel('Prima Componente Principale (PC1)');
ylabel('Seconda Componente Principale (PC2)');
colorbar;
grid on;

% Visualizzazione Scatter Plot 3D - Prime 3 Componenti Principali
if size(score, 2) >= 3  % Verifica che ci siano almeno 3 componenti
    figure;
    scatter3(score(:,1), score(:,2), score(:,3), 50, y, 'filled');
    title('Scatter Plot 3D - Prime 3 Componenti Principali');
    xlabel('Prima Componente Principale (PC1)');
    ylabel('Seconda Componente Principale (PC2)');
    zlabel('Terza Componente Principale (PC3)');
    colorbar;
    grid on;
end
