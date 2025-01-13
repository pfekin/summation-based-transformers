from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

out_size = 200
  
data = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1, shuffle=False)

scaler = StandardScaler()
scaler.fit(x_train)  # Don't cheat - fit only on training data
x_train, x_test = scaler.transform(x_train), scaler.transform(x_test) 

# benchmark standard classifier
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam',  alpha=0.0001, 
                     random_state=1, max_iter=2000)
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print(f"{np.mean(y_test==predicted)}")

# fixed attractor classifier output, convert x_train & x_test
rnd_root = (np.random.random(out_size) - 0.5) / 10.0

y_train_fixed = np.tile(rnd_root , len(y_train)).reshape((len(y_train), out_size))
model_fixed = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam',  alpha=0.0001, 
                     random_state=1, max_iter=2000)
model_fixed.fit(x_train, y_train_fixed) 

new_x_train = model_fixed.predict(x_train)
new_x_test = model_fixed.predict(x_test) 

# output of fixed attractor is input to new classifier
new_model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam',  alpha=0.0001, 
                     random_state=1, max_iter=2000)
new_model.fit(new_x_train, y_train) 

predicted = new_model.predict(new_x_test) 
print(f"{np.mean(y_test==predicted)}")
