import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

import pickle

df = pd.read_csv('steam.csv')
x = df[['pressure','temperature']]
y = df[['vf','vg','uf','ug','hf','hfg','hg','sf','sfg','sg']]

model_filename = 'steam.model'
model = pickle.load(open(model_filename, 'rb'))


x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.05)
accuracy = model.score(x_test, y_test)

print("Source of data: NIST Chemistry WebBook")
print("Please do take note, however, that the predictions are only applicable to saturated state of water")

pressure = float(input('Enter Pressure (MPa): '))
temperature = float(input('Enter Temperature (Celsius): '))

x_predict = [[pressure, temperature]]
y_predict = model.predict(x_predict)
y_df = pd.DataFrame(data=y_predict, columns=['vf','vg','uf','ug','hf','hfg','hg','sf','sfg','sg'])

print('----- For Liquid State -----')
print('Specific Volume (m^3/kg): ',y_df['vf'].item())
print('Internal Energy (kJ/kg): ',y_df['uf'].item())
print('Enthalpy (kJ/kg): ',y_df['hf'].item())
print('Entropy (kJ/kg K): ',y_df['sf'].item())
print('')
print('----- For Vapor State -----')
print('Specific Volume (m^3/kg): ',y_df['vg'].item())
print('Internal Energy (kJ/kg): ',y_df['ug'].item())
print('Enthalpy (kJ/kg): ',y_df['hg'].item())
print('Entropy (kJ/kg K): ',y_df['sg'].item())
print('')
print('Accuracy: ',accuracy)