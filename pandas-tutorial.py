# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=mlcc-prework&hl=en
import numpy as np
import pandas as pd

data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])
columns = ['temperature', 'activity']

df = pd.DataFrame(data=data, columns=columns)
df["adjusted"] = df["activity"] + 2             # Adding a new column to a DataFrame

# some essantial functions
df.head(3)                                      # rows 0, 1, 2
df.iloc[[2]]                                    # row 2
df[1:4]                                         # rows 1, 2, 3
df['temperature']                               # specified column

# Task 1: Create a DataFrame
people = ['Elenor', 'Chidi', 'Tahani', 'Jason']
data = np.random.randint(0, 101, (3, 4))
df = pd.DataFrame(data=data, columns=people)
df['Janet'] = df['Tahani'] + df['Jason']

# Copying a DataFrame

# if we assing df to a new variable with reference. Modifing one of them will change the other.
df_new = df

# if we need an independent copy of that df we use expression below
df_new = df.copy()
