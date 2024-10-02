clc
clear
% Carica il dataset
data = readtable('diabetes(2).csv');

% Separa le features (X) dal target (y)
X = data{:, 1:end-1};
y = data{:, end};

% Step 2-4: Esegui la PCA sul dataset X (le features)
[coeff, score, latent, tsquared, explained, mu] = pca(X);

% Scegli il numero di componenti principali, ad esempio 3
numComponents = 3;

% Riduci il dataset alle prime 3 componenti principali
X_reduced = score(:, 1:numComponents);

% Mostra il dataset ridotto
disp('Dataset ridotto alle prime 3 componenti principali:');
disp(X_reduced);

% Mostra la varianza spiegata da ciascuna componente principale
figure;
pareto(explained);
title('Variance Explained by Principal Components');
xlabel('Principal Component');
ylabel('Variance Explained (%)');

% Cumulative variance explained
cumulativeExplained = cumsum(explained);

% Feature che soddisfano il 95perc
 numComponents95 = find(cumulativeExplained >= 95, 1);

% Numero ridotto di feature per ridurre il dataset
X_reduced_95 = score(:, 1:numComponents95);

% Mostra il numero di componenti usati per spiegare almeno il 95% della varianza
disp(['Numero di componenti usati per spiegare almeno il 95% della varianza: ', num2str(numComponents95)]);
disp('Dataset ridotto per il 95% della varianza spiegata:');
disp(X_reduced_95);

%Visualizzazione Scatter Plot 2D - Prime 2 Componenti Principali
figure;
scatter(X_reduced(:,1), X_reduced(:,2), 50, y, 'filled');
title('Scatter Plot 2D - Prime 2 Componenti Principali');
xlabel('Prima componente principale');
ylabel('Seconda componente principale');
colorbar;
grid on;

%Visualizzazione Scatter Plot 3D - Prime 3 Componenti Principali
figure;
scatter3(X_reduced(:,1), X_reduced(:,2), X_reduced(:,3), 50, y, 'filled');
title('Scatter Plot 3D - Prime 3 Componenti Principali');
xlabel('Prima componente principale');
ylabel('Seconda componente principale');
zlabel('Terza componente principale');
colorbar;
grid on;
