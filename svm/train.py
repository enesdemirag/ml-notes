# Burada dataseti import edip degerleri ve aralarindaki correlasyonu inceliyorum. 

import pickle                       # Modeli kaydetmek icin
import numpy as np
import pandas as pd                 # Dataset icin
from sklearn import svm             # Kullanilan algoritma (Support Vector Machine)

df = pd.read_csv("mammographic_masses_train.csv")

# NaN degerleri (?) cikartiyorum
df = df.dropna()

# Data Range
print(df.describe())
# Correlation
print(df.corr())

# Dataseti inceledim ve Age, Shape, ve Margin verilerinin Severity ile ilgili oldugunu gordum.
# Bu yuzden bu uc veriyi feature'lar olarak sectim.
X = df.loc[ : , ["Age", "Shape", "Margin"]]
Y = df.loc[ : , ["Severity"]]

# Linear model kullandim
model = svm.SVC(kernel="linear").fit(X, Y.values.ravel())
pickle.dump(model, open("model.sav", 'wb'))
