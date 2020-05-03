import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("california-housing-dataset.csv")

df["median_house_value"] /= 500001

longitude = list(df["longitude"])
latitude = list(df["latitude"])
prices = list(df["median_house_value"])

plt.scatter(x = longitude, y = latitude, s = 4, c = prices, cmap='rainbow')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()