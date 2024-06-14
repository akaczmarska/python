#Import bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

#Wywolanie danych
df = pd.read_csv("american_bankruptcy2.csv", sep=';')
print(df.head(3).to_string())

X = df.iloc[:, 2:]
y = df.bankruptcy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Modelu regresji logistycznej
print('\Regresja logistyczna')

model = LogisticRegression(penalty= None, max_iter=200)
# nauka na 80% danych
model.fit(X_train, y_train)
# sprawdzenie modelu na danych testowych
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
precision = tp/(tp+fp)
precision

#zapis modelu do pliku
joblib.dump(model, 'logistic_model.model')
#wczytanie modelu
model_logistic = joblib.load('logistic_model.model')
model_new = joblib.load('logistic_model.model')

#Model KNN
print('\KNN')

model = KNeighborsClassifier(n_neighbors=5, weights='distance')   #, weights='distance')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

results = []
for k in range(1, 50):
    model = KNeighborsClassifier(n_neighbors=k) # , weights='distance')
    model.fit(X_train, y_train)
    results.append(model.score(X_test, y_test))

# plt.scatter(x=range(1, 50), y=results) # wykres punktowy
plt.plot(range(1, 50), results, 'b')   # linia ciągła
plt.show()

#Model drzewo decyzyjne
print('\Drzewo decyzyjne')
model = DecisionTreeClassifier(max_depth=20, min_samples_split=20)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
print(pd.DataFrame(model.feature_importances_, X.columns))




