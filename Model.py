from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('cnn_20_epochs.h5')

# Print the model summary
model.summary()
