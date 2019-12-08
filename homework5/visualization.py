import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NBdf = pd.read_csv('NB_tuning_process.csv')
NBdf.dropna(axis=0, how='any', inplace=True)
x = NBdf['alpha']
y = NBdf['accuracy']
plt.plot(x, y)
plt.title('alpha in Naive Bayes')
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.show()

RFdf = pd.read_csv('RF_tuning_process.csv')
RFdf.dropna(axis=0, how='any', inplace=True)
x = RFdf[RFdf['criterion']=='entropy']['n_estimators']
y1 = RFdf[RFdf['criterion']=='entropy']['accuracy']
y2 = RFdf[RFdf['criterion']=='gini']['accuracy']
plt.plot(x, y1,label = 'entropy')
plt.plot(x, y2,label = 'gini')
plt.title('criterion and n_estimators in Random Forest')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.show()

SVMdf = pd.read_csv('SVM_tuning_process.csv')
SVMdf.dropna(axis=0, how='any', inplace=True)
x = SVMdf[SVMdf['kernel']=='linear']['C']
y1 = SVMdf[SVMdf['kernel']=='linear']['accuracy']
y2 = SVMdf[SVMdf['kernel']=='poly']['accuracy']
y3 = SVMdf[SVMdf['kernel']=='rbf']['accuracy']
y4 = SVMdf[SVMdf['kernel']=='sigmoid']['accuracy']
y5 = SVMdf[SVMdf['kernel']=='precomputed']['accuracy']
plt.plot(x, y1,label = 'linear')
plt.plot(x, y2,label = 'poly')
plt.plot(x, y3,label = 'rbf')
plt.plot(x, y4,label = 'sigmoid')
plt.plot(x, y5,label = 'precomputed')
plt.title('kernel and C in SVM')
plt.xlabel('kernel')
plt.ylabel('accuracy')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
XGBdf = pd.read_csv('XGB_tuning_process.csv')
XGBdf.dropna(axis=0, how='any', inplace=True)
x = XGBdf['subsample']
y = XGBdf['learning_rate']
z = XGBdf['accuracy']
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
ax.set_zlabel('accuracy', fontdict={ 'color': 'red'})
ax.set_ylabel('learning_rate', fontdict={ 'color': 'red'})
ax.set_xlabel('subsample', fontdict={'color': 'red'})
