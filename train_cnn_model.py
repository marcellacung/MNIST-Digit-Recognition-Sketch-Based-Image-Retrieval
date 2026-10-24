import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

print("Membuat model CNN yang lebih baik...")

# Buat model CNN untuk MNIST
def create_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Load dan preprocess MNIST
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize dan reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")

# Buat dan latih model
model = create_cnn_model()
model.summary()

print("\nTraining model CNN (ini akan lebih lama tapi lebih akurat)...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test accuracy: {test_acc:.2%}")

# Simpan model
model.save('mnist_cnn_model.h5')
print("Model CNN disimpan sebagai 'mnist_cnn_model.h5'")

# Simpan juga sebagai pickle (untuk kompatibilitas)
with open('mnist_cnn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model juga disimpan sebagai 'mnist_cnn_model.pkl'")