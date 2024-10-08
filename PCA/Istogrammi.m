clear;
clc;

%Dati --> DatarateAggregato, TempoDiTrasmissione, Energia
Baseline = [76800, 2.49e-6, 1.91e-7];

SHAP1 = [76800, 5.44e-7, 4.18e-8];
SHAP3 = [76800, 1.44e-6, 1.1e-7];

PCA2 = [76800, 3.04e-6, 2.33e-7];
PCA3 = [76800, 4.43e-6, 3.4e-7];

%Label
Base_SHAP = {'Baseline', 'SHAP (1 Feature)', 'SHAP (3 Feature)'};
Base_PCA = {'Baseline', 'PCA (2 Componenti)', 'PCA (3 Componenti)'};
SHAP_PCA = {'PCA (2 Componenti)', 'SHAP (1 Feature)'};

%Dati primo Grafico DataRate
DataRateAggregato = [Baseline(1), PCA2(1), SHAP1(1)];
DataLabel = {'Baseline', 'PCA', 'SHAP'};

%Dati secondo Grafico Tempo di Trasmissione SHAP e Baseline
TempoSHAP = [Baseline(2), SHAP1(2), SHAP3(2)];
%Dati terzo Grafico Tempo di Trasmissione PCA e Baseline
TempoPCA = [Baseline(2), PCA2(2), PCA3(2)];

%Dati quarto Grafico Energia SHAP e Baseline
EnergiaSHAP = [Baseline(3), SHAP1(3), SHAP3(3)];
%Dati quinto Grafico Energia PCA e Baseline
EnergiaPCA = [Baseline(3), PCA2(3), PCA3(3)];

%Dati confronto SHAP3 E PCA3
TempoSHAPCA = [PCA2(2), SHAP1(2)];
EnergiaSHAPCA = [PCA2(3), SHAP1(3)];

%Imposto i colori
coloreBaseline = [0.9, 0.4, 0.1]; % Arancione
colorePCA = [0.1, 0.8, 0.1];      % Verde
coloreSHAP = [0.2, 0.6, 1.0];     % Blu

%1o Istogramma per il DataRate
figure;
b = bar(DataRateAggregato, 'FaceColor', 'flat');
b.CData(1, :) = coloreBaseline;
b.CData(2, :) = colorePCA;
b.CData(3, :) = coloreSHAP;
title('Datarate Aggregato per i tre casi', 'FontSize', 21); % Aumenta la dimensione del font
ylabel('Datarate (Mbps)', 'FontSize', 21); % Aumenta la dimensione del font
set(gca, 'XTickLabel', DataLabel, 'FontSize', 21); % Aumenta la dimensione del font delle etichette degli assi
grid on;

%2o Istogramma Confronto Tempo di Trasmissione Baseline e SHAP
figure;
b = bar(TempoSHAP, 'FaceColor', 'flat');
b.CData(1, :) = coloreBaseline;
b.CData(2, :) = coloreSHAP;
b.CData(3, :) = coloreSHAP;
title('Tempo di Trasmissione tra Baseline e SHAP', 'FontSize', 21);
ylabel('Tempo di trasmissione (s)', 'FontSize', 21);
set(gca, 'XTickLabel', Base_SHAP, 'FontSize', 21);
grid on;

%3o Istogramma Confronto Tempo di Trasmissione Baseline e PCA
figure;
b = bar(TempoPCA, 'FaceColor', 'flat');
b.CData(1, :) = coloreBaseline;
b.CData(2, :) = colorePCA;
b.CData(3, :) = colorePCA;
title('Tempo di Trasmissione tra Baseline e PCA', 'FontSize', 21);
ylabel('Tempo di trasmissione (s)', 'FontSize', 21);
set(gca, 'XTickLabel', Base_PCA, 'FontSize', 21);
grid on;

%4o Istogramma Confronto Energia Baseline e SHAP
figure;
b = bar(EnergiaSHAP, 'FaceColor', 'flat');
b.CData(1, :) = coloreBaseline;
b.CData(2, :) = coloreSHAP;
b.CData(3, :) = coloreSHAP;
title('Energia impiegata tra Baseline e SHAP', 'FontSize', 21);
ylabel('Energia impiegata (j)', 'FontSize', 21);
set(gca, 'XTickLabel', Base_SHAP, 'FontSize', 21);
grid on;

%5o Istogramma Confronto Energia Baseline e PCA
figure;
b = bar(EnergiaPCA, 'FaceColor', 'flat');
b.CData(1, :) = coloreBaseline;
b.CData(2, :) = colorePCA;
b.CData(3, :) = colorePCA;
title('Energia impiegata tra Baseline e PCA', 'FontSize', 21);
ylabel('Energia impiegata (j)', 'FontSize', 21);
set(gca, 'XTickLabel', Base_PCA, 'FontSize', 21);
grid on;

%6o Istogramma Confronto Tempo di Trasmissione tra PCA2 e SHAP1
figure;
b = bar(TempoSHAPCA, 'FaceColor', 'flat');
b.CData(1, :) = colorePCA;
b.CData(2, :) = coloreSHAP;
title('Differenze Tempo di Trasmissione tra PCA e SHAP', 'FontSize', 21);
ylabel('Tempo di Trasmissione (s)', 'FontSize', 21);
set(gca, 'XTickLabel', SHAP_PCA, 'FontSize', 21);
grid on;

%7o Istogramma Confronto Energia impiegata tra PCA2 e SHAP1
figure;
b = bar(EnergiaSHAPCA, 'FaceColor', 'flat');
b.CData(1, :) = colorePCA;
b.CData(2, :) = coloreSHAP;
title('Differenza energia impiegata tra PCA e SHAP', 'FontSize', 21);
ylabel('Energia impiegata (j)', 'FontSize', 21);
set(gca, 'XTickLabel', SHAP_PCA, 'FontSize', 21);
grid on;
