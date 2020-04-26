# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/binary_classification.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=binary_classification_tf2-colab&hl=en#scrollTo=_G6y-XcEmk6r

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Load the dataset
train_df = pd.read_csv("../datasets/california-housing-train.csv")
test_df = pd.read_csv("../datasets/california-housing-test.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index))      # shuffle

# Normalize values using Z-score
train_df_mean = train_df.mean()                                         # mean
train_df_std = train_df.std()                                           # standart deviation
train_df_norm = (train_df - train_df_mean) / train_df_std               # normalize

test_df_mean = test_df.mean()
test_df_std  = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std

# Task 1: Create a binary label
threshold_Z = 1.0 
train_df_norm["is_high"] = (train_df_norm["median_house_value"] > threshold_Z).astype(float)
test_df_norm["is_high"] = (test_df_norm["median_house_value"] > threshold_Z).astype(float)

# Represent features in feature columns
feature_columns = []

median_income = tf.feature_column.numeric_column("median_income")
tr = tf.feature_column.numeric_column("total_rooms")

feature_columns.append(median_income)
feature_columns.append(tr)

feature_layer = layers.DenseFeatures(feature_columns)

feature_layer(dict(train_df_norm))

# Define functions

def create_model(learning_rate, feature_layer, metrics):
    model = tf.keras.models.Sequential()

    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid),)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),                                                   
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=metrics)
    return model        

def train_model(model, dataset, epochs, label_name, batch_size=None, shuffle=True):
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name)) 
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=shuffle)
  
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

def plot_curve(epochs, hist, list_of_metrics):
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.show()

# Hyperparameters
learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "is_high"
classification_threshold = 0.5

METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold),]

# Train
classifier = create_model(learning_rate, feature_layer, METRICS)

epochs, hist = train_model(classifier, train_df_norm, epochs, label_name, batch_size)

list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Test
features = {name:np.array(value) for name, value in test_df_norm.items()}
label = np.array(features.pop(label_name))

classifier.evaluate(x = features, y = label, batch_size=batch_size)

# Task 2: Add precision and recall as metrics
epochs = 20

METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold),
            tf.keras.metrics.Precision(thresholds=classification_threshold, name='precision'),
            tf.keras.metrics.Recall(thresholds=classification_threshold, name="recall"),
]

classifier = create_model(learning_rate, feature_layer, METRICS)

epochs, hist = train_model(classifier, train_df_norm, epochs, label_name, batch_size)

list_of_metrics_to_plot = ['accuracy', "precision", "recall"] 
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Task 3: Summary using Area under the Receiver Operated Characteristic (ROC) Curve (AUC)
epochs = 20

METRICS = [
      tf.keras.metrics.AUC(num_thresholds=100, name='auc'),
]

classifier = create_model(learning_rate, feature_layer, METRICS)

epochs, hist = train_model(classifier, train_df_norm, epochs, label_name, batch_size)

list_of_metrics_to_plot = ['auc'] 
plot_curve(epochs, hist, list_of_metrics_to_plot)