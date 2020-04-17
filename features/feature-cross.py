# Representation with a Feature Cross
# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/representation_with_a_feature_cross.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=representation_tf2-colab&hl=en

# Load modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Settings
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
tf.keras.backend.set_floatx('float32')

# Load
train_df = pd.read_csv("../datasets/california-housing-train.csv")
test_df = pd.read_csv("../datasets/california-housing-test.csv")

# Scale
scale_factor = 1000.0
train_df["median_house_value"] /= scale_factor 
test_df["median_house_value"] /= scale_factor

# Shuffle
train_df = train_df.reindex(np.random.permutation(train_df.index))

# Represent latitude and longitude as floating-point values
feature_columns = []

latitude = tf.feature_column.numeric_column("latitude")
longitude = tf.feature_column.numeric_column("longitude")
feature_columns.append(latitude)
feature_columns.append(longitude)

fp_feature_layer = layers.DenseFeatures(feature_columns)

# Functions
def create_model(learning_rate, feature_layer):
    """Create and compile a simple linear regression model."""
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, dataset, epochs, batch_size, label_name):
    """Feed a dataset into the model in order to train it."""

    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True)
    
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return epochs, rmse

def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.95, rmse.max() * 1.05])
    plt.show()

# Train the model with floating-point representations
learning_rate = 0.05
epochs = 30
batch_size = 100
label_name = 'median_house_value'

model = create_model(learning_rate, fp_feature_layer)
epochs, rmse = train_model(model, train_df, epochs, batch_size, label_name)
plot_the_loss_curve(epochs, rmse)

# Test
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
model.evaluate(x=test_features, y=test_label, batch_size=batch_size)

# Task 1: Represent latitude and longitude in buckets (bins)
# 10 buckets for latitude.
# 10 buckets for longitude.

resolution_in_degrees = 0.4 
feature_columns = []

latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df['latitude'])),
                                    int(max(train_df['latitude'])),
                                    resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                                latitude_boundaries)
feature_columns.append(latitude)

longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df['longitude'])),
                                    int(max(train_df['longitude'])),
                                    resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                longitude_boundaries)
feature_columns.append(longitude)

buckets_feature_layer = layers.DenseFeatures(feature_columns)

# Train the model with bucket representations
learning_rate = 0.04
epochs = 35

model = create_model(learning_rate, buckets_feature_layer)
epochs, rmse = train_model(model, train_df, epochs, batch_size, label_name)
plot_the_loss_curve(epochs, rmse)

# Test
model.evaluate(x=test_features, y=test_label, batch_size=batch_size)

# Task 2: Represent location as a feature cross
resolution_in_degrees = 1.0 

# Create a new empty list that will eventually hold the generated feature column.
feature_columns = []

latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df['latitude'])), int(max(train_df['latitude'])), resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)

longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df['longitude'])), int(max(train_df['longitude'])), resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, longitude_boundaries)

latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

feature_cross_feature_layer = layers.DenseFeatures(feature_columns)

# Train again
learning_rate = 0.04
epochs = 35

model = create_model(learning_rate, feature_cross_feature_layer)
epochs, rmse = train_model(model, train_df, epochs, batch_size, label_name)
plot_the_loss_curve(epochs, rmse)

# Test again
model.evaluate(x=test_features, y=test_label, batch_size=batch_size)


