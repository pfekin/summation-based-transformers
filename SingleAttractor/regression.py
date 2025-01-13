# importing modules and packages 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

out_size = 200 # random vector output size

# importing data 
df = pd.read_csv('Real-estate.csv') 
df.drop('No', inplace=True, axis=1) 

print(df.head()) 

print(df.columns) 

# plotting a scatterplot 
sns.scatterplot(x='X4 number of convenience stores', 
				y='Y house price of unit area', data=df) 

# creating feature variables 
x = df.drop('Y house price of unit area', axis=1) 
y = df['Y house price of unit area'] 

print(x) 
print(y) 

# creating train and test sets 
x_train, x_test, y_train, y_test = train_test_split( 
	x, y, test_size=0.3, random_state=1) 
    
scaler = StandardScaler()
scaler.fit(x_train)  # Don't cheat - fit only on training data
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

# creating a regression model 
model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='sgd',  alpha=0.0001, 
                     random_state=1, max_iter=4000)

# fitting the model 
model.fit(x_train, y_train) 

# making predictions 
predictions = model.predict(x_test) 

# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

# fixed attractor regressor output
rnd_root = (np.random.random(out_size) - 0.5) / 10.0
y_train_fixed = np.tile(rnd_root , len(y_train)).reshape((len(y_train), out_size))

model_fixed = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='sgd',  alpha=0.0001, 
                     random_state=2, max_iter=4000)
model_fixed.fit(x_train, y_train_fixed) 

new_x_train = model_fixed.predict(x_train)
new_x_test = model_fixed.predict(x_test) 

# output of fixed attractor is input to new regressor
new_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='sgd',  alpha=0.0001, 
                     random_state=1, max_iter=4000)
new_model.fit(new_x_train, y_train) 

predictions = new_model.predict(new_x_test) 
print('attractor mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('attractor mean_absolute_error : ', mean_absolute_error(y_test, predictions))


