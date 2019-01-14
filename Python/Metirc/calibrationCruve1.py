# coding=utf-8
import matplotlib.pyplot as plt
import random
import csv
import numpy as np

path = '/media/zzr/Data/git/Classification_featureselection/PNET_CT/DC_testing1.csv'
data = []
with open(path) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        data.append([float(i) for i in row])

data = np.array(data)
label = data[:, 0]
predict = data[:, 1]
threshold = 0.3241
#
x = range(0, 99)
np.random.shuffle(x)
group = 8
GT_pro = []
Pre_pro = []
num = np.array(x).shape[0]/group
for i in xrange(group):
    start = i*num
    end = start + num
    idx = x[start:end]
    predict_ = predict[idx]
    label_ = label[idx]
    Pre_num = predict_[predict_ > threshold].shape[0]/np.float(num)
    GT_num = label_[label_ == 1].shape[0]/np.float(num)
    GT_pro.append(GT_num)
    Pre_pro.append(Pre_num)

ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=2)
ax1.plot([0, 1], [0, 1], "k:")
ax1.scatter(Pre_pro, GT_pro)
z = np.polyfit(Pre_pro, GT_pro, 2)
p1 = np.poly1d(z)
print p1
zcalue = p1(Pre_pro)
zcalue = list(zcalue[np.argsort(Pre_pro)])
zcalue.insert(0, 0.)
Pre_pro = sorted(Pre_pro)
Pre_pro.insert(0, 0.)
plt.plot(Pre_pro, zcalue)
plt.show()