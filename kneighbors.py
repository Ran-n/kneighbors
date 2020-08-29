#! /usr/bin/python
#+ Autor:	Ran#
#+ Creado:	29/08/2020 13:26:24
#+ Editado:	29/08/2020 13:42:01

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.datasets import make_blobs
#import mglearn


while True:
    vecinhos = input('> Cantos veciños?: ')
    if vecinhos.isdigit():
        vecinhos = int(vecinhos)
        break

# collemos os datos e as etiquetas
#X, y = mglearn.datasets.make_forge()
X, y = make_blobs()

# partimos os datos entre entrenamento e testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# candos veciños queremos mirar
clf = knc(n_neighbors=vecinhos)

clf.fit(X_train, y_train)

knc(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights='uniform')

print('Predicción: {}'.format(clf.predict(X_test)))

print('Puntuación : {}'.format(clf.score(X_test, y_test)))
