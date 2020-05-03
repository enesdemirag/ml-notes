# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/multi-class_classification_with_MNIST.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=multiclass_tf2-colab&hl=en#scrollTo=9n9_cTveKmse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
np.set_printoptions(linewidth = 200)

# Load the dataset
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

# View the dataset
plt.imshow(x_train[1864])   # sample
x_train[1864][10][16]       # pixel value

# Task 1: Normalize feature values
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

# Define a plotting function
def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    plt.legend()

# Create Model
# 256 nodes in first layer
# dropout 0.4
def create_model(learning_rate):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))     
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="sparse_categorical_crossentropy", metrics=['accuracy'])
  
    return model    

def train_model(model, train_features, train_label, epochs, batch_size=None, validation_split=0.1):
    history = model.fit(x=train_features,
                        y=train_label,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True, 
                        validation_split=validation_split)
    
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

# Train
learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

multi_classifier = create_model(learning_rate)

epochs, hist = train_model(multi_classifier, x_train_normalized, y_train, epochs, batch_size, validation_split)

list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

multi_classifier.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)