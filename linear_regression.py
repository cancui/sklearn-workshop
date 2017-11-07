import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

'''
Create a linear regresssion object. This contains the functions for fitting and predicting,
and the parameters that are learned.

We'll use this to predict core body tempereature from skin temperature - I fabricated the problem
''' 
regressor = LinearRegression()

'''
Create object responsible for creating higher order polynomial features from original features
'''
third_degree_features = PolynomialFeatures(degree=3)

'''
Fabricate the data we'll use (features and labels)
feature: skin_temperature
label: core_temperature
'''
x = np.arange(-5,5,0.05)
skin_temperature = x*1.5+22
core_temperature = 1 / (1 + np.exp(-x))*3 + 35 + np.random.normal(0, 0.1, x.shape[0])

'''
sklearn provides a function to randomly split data into training and test sets
'''
skin_temperature_train, skin_temperature_test, core_temperature_train, core_temperature_test = train_test_split(skin_temperature, core_temperature)

plt.figure()
plt.scatter(skin_temperature_train, core_temperature_train, 3)
plt.xlabel('Skin Temperature (C)')
plt.ylabel('Core Temperature (C)')

'''
Training stage: fit linear regression to the data, where the features are only the raw data (only first order features)
'''
features = skin_temperature_train.reshape(-1,1)
regressor.fit(features, core_temperature_train)

'''
Get the line that was fitted to the data, and plot
'''
m = regressor.coef_
b = regressor.intercept_
line = m * skin_temperature + b
plt.plot(skin_temperature, line, color='orange')

'''
Create 1st, 2nd, and 3rd order features from our raw data, and train again to fit linear regression to these
'''
features = third_degree_features.fit_transform(skin_temperature_train.reshape(-1,1))
regressor.fit(features, core_temperature_train)

'''
Plot our new model
'''
_, c1, c2, c3 = regressor.coef_
m = regressor.coef_
b = regressor.intercept_
line = c1 * skin_temperature**1 + c2 * skin_temperature**2 + c3 * skin_temperature**3 + b
plt.plot(skin_temperature, line, color='red')

'''
Plot the training set
'''
plt.figure()
plt.scatter(skin_temperature_train, core_temperature_train, 3)
plt.xlabel('Skin Temperature (C)')
plt.ylabel('Core Temperature (C)')

'''
Test model by inputting some skin temperature data. Model outputs predictions for core temperatures
'''
input_skin_temps = np.asarray([24.1, 22.5, 21.3]).reshape(-1,1)
input_feature = third_degree_features.transform(input_skin_temps)
model_prediction = regressor.predict(input_feature)

np.set_printoptions(precision=1)
print('\n##### Results #####')
print('Input skin temperatures:  {}'.format(input_skin_temps.reshape(-1)))
print('Output core temperatures: {}'.format(model_prediction))


'''
Test model quantitatively by getting the error of the model on a separeate "test" dataset
'''
test_features = third_degree_features.transform(skin_temperature_test.reshape(-1,1))
test_predicions = regressor.predict(test_features)

error = mean_squared_error(core_temperature_test, test_predicions)
print('Mean squared error is {}'.format(error))

plt.show()