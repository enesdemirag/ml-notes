import pickle                       # Modeli import etmek icin
import numpy as np
import pandas as pd                 # Dataset icin
from sklearn import svm             # Kullanilan algoritma (Support Vector Machine)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt     # Visualize etmek icin
from mpl_toolkits.mplot3d import Axes3D

model = pickle.load(open("model.sav", 'rb'))

df = pd.read_csv("mammographic_masses_test.csv")
df = df.dropna()

X = df.loc[ : , ["Age", "Shape", "Margin"]]
Y = df["Severity"]

prediction = model.predict(X)
# print(prediction)
print(accuracy_score(Y, prediction))

# Gorsellik acisindan modeli featurelar ile 3D uzayda gorsellestirdim
age = list(df["Age"])
shape = list(df["Shape"])
margin = list(df["Margin"])
prediction = list(prediction)
real = list(Y)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.scatter(age, shape, margin, s=30, c=prediction)
ax2.scatter(age, shape, margin, s=30, c=real)

plt.tight_layout()
plt.show()
