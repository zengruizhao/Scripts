# coding=utf-8
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import os
import csv
import numpy as np

path = '/media/zzr/Data/git/Classification_featureselection/PNET_CT/DC_2cm1.csv'
data = []
with open(path) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        data.append([float(i) for i in row])

data = np.array(data)
label = data[:, 0]
predict = data[:, 1]
# print label
# print predict
# print 'brier score: ', brier_score_loss(label, predict)
prob_true, prob_pred = calibration_curve(label, predict, normalize=True, n_bins=5)
fig = plt.figure()
ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=2)
ax1.plot(prob_pred, prob_true)
ax1.plot([0, 1], [0, 1], "k:")
plt.show()
