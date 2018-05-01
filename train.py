import pandas as pd
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression


import pickle

df = pd.read_csv('steam.csv')

x = df[['pressure', 'temperature']]
y = df[['vf','vg','uf','ug','hf','hfg','hg','sf','sfg','sg']]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.05)# 20% data will be used as testing data

#model = svm.SVR(kernel='linear')
model = LinearRegression()
model.fit(x_train, y_train)

model_filename = 'steam.model'
pickle.dump(model, open(model_filename, 'wb'))