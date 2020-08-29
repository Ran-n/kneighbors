#! /usr/bin/python
#+ Autor:	Ran#
#+ Creado:	29/08/2020 18:57:11
#+ Editado:	29/08/2020 18:57:11

from sklearn.neighbors import KNeighborsRegressor as knr
from sklearn.model_selection import train_test_split
import mglearn

X,y = mglearn.datasets.make_wave(n_samples=400)

# split the wave dataset into a training anda test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Instantiate the model, set the number of neighbors to consider to 3
reg = knr(n_neighbors=3)

# Fit the model using the training data and training targets
reg.fit(X_train, y_train)

knr(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights='uniform')

prediccion = reg.predict(X_test)
print('- Predicción: ', prediccion)

ben = []
for ele in zip(prediccion, y_test):
    ben.append(ele[0]==ele[1])

print('\n- Valores acertados ou errados: ', ben)

print('\n- Calificación: ', reg.score(X_test, y_test))
