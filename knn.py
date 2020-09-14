import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv('car.data')
predict = 'class'

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying'])) 
maint = le.fit_transform(list(data['maint'])) 
door = le.fit_transform(list(data['door'])) 
persons = le.fit_transform(list(data['persons'])) 
lug_boot = le.fit_transform(list(data['lug_boot'])) 
safety = le.fit_transform(list(data['safety'])) 
clss = le.fit_transform(list(data[predict]))


X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(clss)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

KNN_Model = KNeighborsClassifier(n_neighbors = 9)
KNN_Model.fit(x_train, y_train)
acc = KNN_Model.score(x_test, y_test)
print(acc)

predicted = KNN_Model.predict(x_test)
names = le.classes_

for i in range(len(predicted)):
    print('Prediction: ', names[predicted[i]],'|', "Data: ", x_test[i], '|', "Actual: ", names[y_test[i]])