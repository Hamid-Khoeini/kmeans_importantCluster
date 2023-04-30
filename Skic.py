#Seeded Kmeans important cluster

import pandas as pd

#load data
df = pd.read_csv('iris.csv')
#add columns name
df.columns = ['x1','x2','x3','x4','class_name']
#add numerical label
df['class_int'] = pd.Categorical(df.iloc[:,-1]).codes
#get the number of classes
nuOfclass = set(df.iloc[:,-1])



