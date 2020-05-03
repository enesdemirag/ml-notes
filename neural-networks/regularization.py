# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/intro_to_neural_nets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=intro_to_nn_tf2-colab&hl=en

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

train_df = pd.read_csv("../datasets/california-housing-train.csv")
test_df = pd.read_csv("../datasets/california-housing-test.csv")

# Shuffle the examples
train_df = train_df.reindex(np.random.permutation(train_df.index))

# Normalize values
# Calculate the Z-scores of each column in the training set:
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean) / train_df_std

# Calculate the Z-scores of each column in the test set.
test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean) / test_df_std

# Represent data
feature_columns = []

resolution_Z = 0.3

latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")

latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])), 
                                    int(max(train_df_norm['latitude'])), 
                                    resolution_Z))
longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])), 
                                    int(max(train_df_norm['longitude'])), 
                                    resolution_Z))

latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, longitude_boundaries)

latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
median_income = tf.feature_column.numeric_column("median_income")
population = tf.feature_column.numeric_column("population")

feature_columns.append(crossed_feature)
feature_columns.append(median_income)
feature_columns.append(population)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Plot Function
def plot_the_loss_curve(epochs, mse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min()*0.95, mse.max() * 1.03])
    plt.show()

# Deep Neural Network Model with L1 Regularization
def create_model_l1(learning_rate, my_feature_layer):
    model = tf.keras.models.Sequential()
    model.add(my_feature_layer)
    model.add(tf.keras.layers.Dense(units=10, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                                  name='Hidden1'))
    model.add(tf.keras.layers.Dense(units=6, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                                  name='Hidden2'))
    model.add(tf.keras.layers.Dense(units=1, name='Output'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

# Deep Neural Network Model with L1 Regularization
def create_model_l2(learning_rate, my_feature_layer):
    model = tf.keras.models.Sequential()
    model.add(my_feature_layer)
    model.add(tf.keras.layers.Dense(units=10, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                  name='Hidden1'))
    model.add(tf.keras.layers.Dense(units=6, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                  name='Hidden2'))
    model.add(tf.keras.layers.Dense(units=1, name='Output'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

# Deep Neural Network Model with L1 Regularization
def create_model_dropout(learning_rate, my_feature_layer):
    model = tf.keras.models.Sequential()
    model.add(my_feature_layer)
    model.add(tf.keras.layers.Dense(units=10, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                                  name='Hidden1'))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Dense(units=6, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                                  name='Hidden2'))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Dense(units=1, name='Output'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

# Define a training function
def train_model(model, dataset, epochs, label_name, batch_size=None):
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True) 
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]
    return epochs, mse

# Train Neural Network Model
learning_rate = 0.01
batch_size = 1000

label_name = "median_house_value"

l1 = create_model_l1(learning_rate, feature_layer)
l2 = create_model_l2(learning_rate, feature_layer)
dropout = create_model_dropout(learning_rate, feature_layer)

epochs, mse = train_model(l1, train_df_norm, 20, label_name, batch_size)
plot_the_loss_curve(epochs, mse)

epochs, mse = train_model(l2, train_df_norm, 20, label_name, batch_size)
plot_the_loss_curve(epochs, mse)

epochs, mse = train_model(dropout, train_df_norm, 20, label_name, batch_size)
plot_the_loss_curve(epochs, mse)

test_features = {name:np.array(value) for name, value in test_df_norm.items()}
test_label = np.array(test_features.pop(label_name))

# Test models and see the best one (smallest loss should be L2)
l1.evaluate(x = test_features, y = test_label, batch_size=batch_size)
l2.evaluate(x = test_features, y = test_label, batch_size=batch_size)
dropout.evaluate(x = test_features, y = test_label, batch_size=batch_size)