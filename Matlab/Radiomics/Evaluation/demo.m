clear;clc;
load fisheriris
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4);
y = species(inds);
Y(strcmp(y, 'versicolor')) = 1;
Y(strcmp(y, 'virginica')) = 0;
mdl = fitglm(X,Y','Distribution','binomial');
probs = predict(mdl, X);
[HLstat, HLp, contingencyM] = HLtest1([probs, Y'], 5); % set degree of freedom
% The larger HLp(P value), the better the calibration of the model.
x = contingencyM(:, 3);
y = contingencyM(:, 4);
%% curve fit
% y1 = polyfit(x, y, 2);
% y = polyval(y1, x);
%%
plot(x/ceil(max(x)), y/ceil(max(y)), 'LineWidth', 2);hold on;
plot([0, 1], [0, 1], '--', 'LineWidth', 2);
axis([0 1 0 1]);
title('Calibration Curve', 'FontSize',16);
xlabel('Radiomics-Predicted Probability', 'FontSize',16);
ylabel('Actual Rate of Grade 2', 'FontSize',16);hold off;
% Decision Curve
figure;DECISIONCURVE(probs, Y');