#! /usr/bin/python
#+ Autor:	Ran#
#+ Creado:	29/08/2020 13:26:24
#+ Editado:	29/08/2020 14:29:31

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt


while True:
    vecinhos = input('> Cantos veciños?: ')
    if vecinhos.isdigit():
        vecinhos = int(vecinhos)
        break

# collemos os datos e as etiquetas
X, y = mglearn.datasets.make_forge()
#X, y = make_blobs()

# partimos os datos entre entrenamento e testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# candos veciños queremos mirar
clf = knc(n_neighbors=vecinhos)

clf.fit(X_train, y_train)

knc(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights='uniform')

print('Predicción: {}'.format(clf.predict(X_test)))

print('Puntuación : {}'.format(clf.score(X_test, y_test)))

#fig, axes = plt.subplots(1, 3, figsize=(10, 3))

#for n_neighbors, ax in zip([1, 3, 9], axes):
#    clf = knc(n_neighbors=n_neighbors).fit(X, y)
#    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#    ax.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
#    ax.set_title("%d neighbor(s)" % n_neighbors)
#plt.show()
