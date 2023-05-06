#Seeded Kmeans important cluster

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data
#add columns name
df = pd.read_csv('iris.csv',names=['x1','x2','x3','x4','class_name'])

#add numerical label
df['class_int'] = pd.Categorical(df.iloc[:,-1]).codes

#show head and tail of data
print('\n')
print('Head of dataset')
print(df.head())
print('\n')
print('Tail of datset')
print(df.tail())

#split data to train and test
train = df.sample(frac=0.7, random_state=np.random.RandomState())
test = df.loc[~df.index.isin(train.index)]

#split data to classes
class1 = train.loc[df.class_int == 0]
class2 = train.loc[df.class_int == 1]
class3 = train.loc[df.class_int == 2]

#find mean of classes
meanFeaturesClass1 = class1.iloc[:,[0,1,2,3]].mean(axis='index').tolist()
meanFeaturesClass2 = class2.iloc[:,[0,1,2,3]].mean(axis='index').tolist()
meanFeaturesClass3 = class3.iloc[:,[0,1,2,3]].mean(axis='index').tolist()
dfmeans = pd.DataFrame(np.array([meanFeaturesClass1,meanFeaturesClass2,meanFeaturesClass3]), \
                       columns = ['x1','x2','x3','x4'])
print('\n\n\n\n\n')
print('means of classes')
print(dfmeans)

#show data and centroids
plt.scatter(df.x1, df.x2 , c=list(df['class_int']))
plt.scatter(dfmeans.x1, dfmeans.x2 , s=120, c='Red')
plt.show()
