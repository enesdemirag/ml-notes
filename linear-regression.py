# Linear Regression with Synthetic Data
# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/linear_regression_with_synthetic_data.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=linear_regression_synthetic_tf2-colab&hl=en

# Import relevant modules
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# Define functions that build and train a model
def build_model(learning_rate):
    """Create and compile a simple linear regression model."""
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def train_model(model, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""
    history = model.fit(x=feature,
                        y=label,
                        batch_size=None,
                        epochs=epochs)

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse

# Define plotting functions

def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against the training feature and label."""
    plt.xlabel("feature")
    plt.ylabel("label")

    plt.scatter(feature, label)

    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')
    plt.show()

def plot_the_loss_curve(epochs, rmse):
    """Plot the loss curve, which shows loss vs. epoch."""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()

# Define the dataset

features = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
labels   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

# Specify the hyperparameters

learning_rate = 0.14
epochs = 100
batch_size = 10

model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(model,
                                                        features,
                                                        labels,
                                                        epochs,
                                                        batch_size)

plot_the_model(trained_weight, trained_bias, features, labels)
plot_the_loss_curve(epochs, rmse)


# Linear Regression with a Real Dataset
# The dataset for this exercise is based on 1990 census data from California.
# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/linear_regression_with_a_real_dataset.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=linear_regression_real_tf2-colab&hl=en#scrollTo=JJZEgJQSjyK4

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Import the dataset.
training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Scale the label.
training_df["median_house_value"] /= 1000.0 # Scaling label values is not essential.

# Get statistics on the dataset.
print(training_df.describe())
# describe() fucntion shows count, mean, std, min, 25%, 50%, 75%, max values of every feature.

def train_model_df(model, df, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse

def plot_the_model_df(trained_weight, trained_bias, feature, label):
    """Plot the trained model against the training feature and label."""
    plt.xlabel(feature)
    plt.ylabel(label)
    
    random_examples = training_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    x0 = 0
    y0 = trained_bias
    x1 = 10000
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')
    plt.show()

# Hyperparameters
learning_rate = 0.01
epochs = 30
batch_size = 30

feature = "total_rooms"  # only one feature used
label="median_house_value"

model = None

# Training
model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model_df(model,
                                            training_df, 
                                            feature,
                                            label,
                                            epochs,
                                            batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias)

plot_the_model_df(weight, bias, feature, label)
plot_the_loss_curve(epochs, rmse)

# Prediction
def predict_house_values(n, feature, label):
    """Predict house values based on a feature."""

    batch = training_df[feature][10000:10000 + n]
    predicted_values = model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(n):
        print ("%5.0f %6.0f %15.0f" % (training_df[feature][i],
                                        training_df[label][i],
                                        predicted_values[i][0]))

predict_house_values(10, feature, label)

# Task 3: Try a different feature
# using population as the feature instead of total_rooms

feature = "population"

learning_rate = 0.05
epochs = 18
batch_size = 3

model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model_df(model,
                                            training_df, 
                                            feature,
                                            label,
                                            epochs,
                                            batch_size)
plot_the_model_df(weight, bias, feature, label)
plot_the_loss_curve(epochs, rmse)

predict_house_values(15, feature, label)

# Task 4: Define a synthetic feature
# total_rooms and population were not useful features
# perhaps block density relates to median house value
# ratio of total_rooms to population might have some predictive power

training_df["rooms_per_person"] = training_df["total_rooms"] / training_df["population"]
feature = "rooms_per_person"

learning_rate = 0.06
epochs = 24
batch_size = 30

model = build_model(learning_rate)
weight, bias, epochs, mae = train_model_df(model,
                                            training_df, 
                                            feature,
                                            label,
                                            epochs,
                                            batch_size)

plot_the_loss_curve(epochs, mae)

predict_house_values(15, feature, label)

# Task 5. Find feature(s) whose raw values correlate with the label
# A correlation matrix indicates how each attribute's raw values relate to the other attributes' raw values.

# Generate a correlation matrix.
print(training_df.corr())

# The "median_income" correlates 0.7 with the label

feature = "median_income"

learning_rate = 0.05
epochs = 20
batch_size = 30

model = build_model(learning_rate)
weight, bias, epochs, mae = train_model_df(model,
                                            training_df, 
                                            feature,
                                            label,
                                            epochs,
                                            batch_size)

plot_the_loss_curve(epochs, mae)

predict_house_values(15, feature, label)