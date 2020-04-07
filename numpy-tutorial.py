# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/numpy_ultraquick_tutorial.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=mlcc-prework&hl=en
import numpy as np

# np.array()
v = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])          # a vector
A = np.array([[6, 5], [11, 7], [4, 8]])                         # a matrix
s = np.arange(5, 12)                                            # range from 5 to 11

# random
x = np.random.randint(low=50, high=101, size=(6))               # 6 integers between 50 - 100
y = np.random.random([6])                                       # 6 floats between 0.0 - 1.0

# operators
z = x + 2                                                       # adds 2 to every component
t = x * 3                                                       # scales every component with 3
k = x + y                                                       # shapes should be same

# Task 1: Create a Linear Dataset
feature = np.arange(6, 21)
label = (3 * feature) + 4

# Task 2: Add Some Noise to the Dataset
noise = np.random.random([len(label)])                          # noise to make it more realistic
label = label + noise

print(feature)
print(label)