clear;clc;
% load aucCombination
% fprCombination = fprTest;
% tprCombination = tprTest;
% load aucT1
% fprT1 = fprTest;
% tprT1 = tprTest;
% load aucT2
% fprT2 = fprTest;
% tprT2 = tprTest;
load aucCombination2cm
fprCombination2cm = fprTest;
tprCombination2cm = tprTest;
load aucT12cm
fprT12cm = fprTest;
tprT12cm = tprTest;
load aucT22cm
fprT22cm = fprTest;
tprT22cm = tprTest;
% h = plot(fprCombination, tprCombination, 'r');hold on;set(h, 'LineStyle', '-');
% h = plot(fprT1, tprT1, 'g');hold on;set(h, 'LineStyle', '-');
% h = plot(fprT2, tprT2, 'b');hold on;set(h, 'LineStyle', '-');
h = plot(fprCombination2cm, tprCombination2cm, 'r');hold on;set(h, 'LineStyle', '-');
h = plot(fprT12cm, tprT12cm, 'g');hold on;set(h, 'LineStyle', '-');
h = plot(fprT22cm, tprT22cm, 'b');hold off;set(h, 'LineStyle', '-');