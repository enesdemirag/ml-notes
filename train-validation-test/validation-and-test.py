# Validation and Test Sets
# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/validation_and_test_sets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=validation_tf2-colab&hl=en

# Import modules
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

train_df = pd.read_csv("../datasets/california-housing-train.csv")
test_df = pd.read_csv("../datasets/california-housing-test.csv")

# Scale the label values
scale_factor = 1000.0

# Training set's label.
train_df["median_house_value"] /= scale_factor 

# Test set's label
test_df["median_house_value"] /= scale_factor

# Functions
def build_model(learning_rate):
    """Create and compile a simple linear regression model."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, df, feature, label, epochs, batch_size=None, validation_split=0.1):
    """Feed a dataset into the model in order to train it."""
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split)
    
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    
    epochs = history.epoch
  
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return epochs, rmse, history.history

def plot_the_loss_curve(epochs, mae_training, mae_validation):
    """Plot a curve of loss vs. epoch."""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    plt.legend()
  
    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)
   
    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()

# Task 1: Experiment with the validation split

learning_rate = 0.08
epochs = 30
batch_size = 100

validation_split = 0.2 # 20% of the training set

feature="median_income"
label="median_house_value"

model = None
model = build_model(learning_rate)

epochs, rmse, history = train_model(model,
                                    train_df,
                                    feature, 
                                    label,
                                    epochs,
                                    batch_size, 
                                    validation_split)

plot_the_loss_curve(epochs,
                    history["root_mean_squared_error"], 
                    history["val_root_mean_squared_error"])

# Task 2: Determine why the loss curves differ

print(train_df.head(n=10))
# We can see that the original training set is sorted by longitude.

# Task 3. Fix the problem
# Shuffle the examples in the training set before splitting

epochs = 70
model = None
shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

model = build_model(learning_rate)
epochs, rmse, history = train_model(model,
                                    shuffled_train_df,
                                    feature, 
                                    label,
                                    epochs,
                                    batch_size, 
                                    validation_split)

plot_the_loss_curve(epochs,
                    history["root_mean_squared_error"], 
                    history["val_root_mean_squared_error"])

# Task 4: Use the Test Dataset to Evaluate Your Model's Performance

x_test = test_df[feature]
y_test = test_df[label]

results = model.evaluate(x_test, y_test, batch_size=batch_size)

# The root mean squared error of all three sets should be similar.