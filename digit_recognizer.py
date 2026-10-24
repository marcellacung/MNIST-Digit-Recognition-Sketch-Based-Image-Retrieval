import numpy as np
from PIL import Image
import tensorflow as tf
import io

class DigitRecognizer:
    def __init__(self):
        # Load a pre-trained MNIST model
        self.model = self.create_model()
        
    def create_model(self):
        # Simple CNN model for MNIST
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Load pre-trained weights (we'll train it in the next step)
        # For now, we'll use a model trained on MNIST
        # You can download pre-trained weights or train it yourself
        return model
    
    def preprocess_image(self, image_data):
        """Convert base64 image to preprocessed numpy array"""
        # Remove header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 (MNIST size)
        image = image.resize((28, 28))
        
        # Invert colors (MNIST has white digits on black background)
        image = Image.fromarray(255 - np.array(image))
        
        # Normalize to [0, 1]
        image_array = np.array(image) / 255.0
        
        # Reshape for model (1, 28, 28, 1)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    
    def predict_digit(self, image_data):
        """Predict digit from image data"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get the predicted digit and confidence
            predicted_digit = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_digit])
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {"digit": int(i), "confidence": float(predictions[0][i])}
                for i in top_3_indices
            ]
            
            return {
                "digit": int(predicted_digit),
                "confidence": confidence,
                "all_predictions": top_3_predictions,
                "success": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def train_model(self, epochs=5):
        """Train the model on MNIST dataset"""
        print("Loading MNIST dataset...")
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Reshape and normalize
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
        
        print(f"Training on {len(x_train)} samples...")
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            batch_size=64,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        return history

# Create a global recognizer instance
recognizer = DigitRecognizer()