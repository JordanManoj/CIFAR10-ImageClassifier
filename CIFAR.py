import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense
from keras.models import Sequential

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix

# DOWNLOADING AND LOADING THE DATA
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

#DATA VISUALIZATION

# Define the labels of the dataset
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

# Define the dimensions of the plot grid 
W_grid = 10
L_grid = 10

# Create the figure and axes objects for the grid
fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
axes = axes.ravel()  # Flatten the 10 x 10 matrix into a 100 element array

n_train = len(X_train)  # Get the length of the train dataset

# Plot images in the grid
for i in np.arange(0, W_grid * L_grid):  # Create evenly spaced variables 
    # Select a random number
    index = np.random.randint(0, n_train)
    # Read and display an image with the selected index    
    axes[i].imshow(X_train[index])
    label_index = int(y_train[index])
    axes[i].set_title(labels[label_index], fontsize=8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()

# Define the labels of the dataset
classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Get unique classes and their counts
classes, counts = np.unique(y_train, return_counts=True)

# Plot the class distribution
plt.barh(classes_name, counts)
plt.title('Class Distribution in Training Set')
plt.xlabel('Number of Images')
plt.ylabel('Class')
plt.show()

# Get unique classes and their counts for the testing set
classes, counts = np.unique(y_test, return_counts=True)

# Plot the class distribution in the testing set
plt.barh(classes_name, counts)
plt.title('Class Distribution in Testing Set')
plt.xlabel('Number of Images')
plt.ylabel('Class')
plt.show()

# Scale the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transforming the target variable into one-hot encoding
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)


#MODEL BUILDING

INPUT_SHAPE = (32, 32, 3)
KERNEL_SIZE = (3, 3)
model = Sequential()

# Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
# Pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))
# Dropout layers
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=2)

batch_size = 32
data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(X_train, y_cat_train, batch_size)
steps_per_epoch = X_train.shape[0] // batch_size

history = model.fit(train_generator, 
                    epochs=10,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=(X_test, y_cat_test), 
                    callbacks=[early_stop],
                   )

# Plotting the training history
plt.figure(figsize=(12, 16))

# Plot Loss
plt.subplot(4, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='val_Loss')
plt.title('Loss Function Evolution')
plt.legend()

# Plot Accuracy
plt.subplot(4, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy Function Evolution')
plt.legend()

# Plot Precision
plt.subplot(4, 2, 3)
plt.plot(history.history['precision'], label='precision')
plt.plot(history.history['val_precision'], label='val_precision')
plt.title('Precision Function Evolution')
plt.legend()

# Plot Recall
plt.subplot(4, 2, 4)
plt.plot(history.history['recall'], label='recall')
plt.plot(history.history['val_recall'], label='val_recall')
plt.title('Recall Function Evolution')
plt.legend()

# Displaying the plots
plt.tight_layout()
plt.show()

# Evaluating the model
evaluation = model.evaluate(X_test, y_cat_test)
print(f'Test Accuracy : {evaluation[1] * 100:.2f}%')

# Predictin the labels for the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Displaying the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Ploting confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(xticks_rotation='vertical', ax=ax, cmap='summer')

plt.show()

print(classification_report(y_test, y_pred, target_names=labels))

# Display a specific image from the test set(DEER)
my_image = X_test[200]
plt.imshow(my_image)
plt.title('Image 200')
plt.show()

# Check the actual label (DEER)
print(f"Actual label for image 200: {labels[y_test[200][0]]}")

# Predict the label for the specific image (DEER)
pred_200 = np.argmax(model.predict(my_image.reshape(1, 32, 32, 3)))
print(f"The model predicts that image 200 is: {labels[pred_200]}")

# Define the labels of the dataset
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

# Define the dimensions of the plot grid 
W_grid = 5
L_grid = 5

# Create the figure and axes objects for the grid
fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
axes = axes.ravel()  # Flatten the 5 x 5 matrix into a 25 element array

n_test = len(X_test)  # Get the length of the test dataset

# Plot images in the grid
for i in np.arange(0, W_grid * L_grid):  # Create evenly spaced variables 
    # Select a random number
    index = np.random.randint(0, n_test)
    # Read and display an image with the selected index    
    axes[i].imshow(X_test[index])
    label_index = int(y_pred[index])  # Use the predicted labels
    axes[i].set_title(f"{index}: {labels[label_index]}", fontsize=8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()

#MODEL EVALUVATION


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[int(true_label)]),
                                         color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, int(true_label[i])
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Assuming you have loaded your model and made predictions
predictions = model.predict(X_test)

# Define the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define the number of rows and columns for the grid of images
num_rows = 8
num_cols = 5
num_images = num_rows * num_cols

# Create the figure and set the size
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

# Plot each image along with its predicted label and prediction array
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], y_test, X_test, class_names)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], y_test)

plt.tight_layout()
plt.show()

#DENSENET MODEL

model = Sequential()
base_model = DenseNet121(input_shape=(32, 32, 3), include_top=False, weights='imagenet', pooling='avg')
model.add(base_model)
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
r = model.fit(train_generator, 
              epochs=100,
              steps_per_epoch=steps_per_epoch,
              validation_data=(X_test, y_cat_test), 
#               callbacks=[early_stop],
             )

model_path = 'C:/Users/jorda/OneDrive/Desktop/CodeTech/DEEP LEARNING FOR IMAGE RECOGNITION/cnn_20_epochs.h5'
model.save(model_path)
