# train_fc_model.py - FIXED VERSION
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

print("Training Fully Connected Model for MNIST...")

# Load MNIST data
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X / 255.0  # Normalize
y = y.astype(int)

# Use smaller dataset for faster training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=8000, test_size=2000, random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create Fully Connected model (MLP)
print("\nCreating MLP (Fully Connected) model...")
fc_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=128,
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=20,  # 20 epochs
    shuffle=True,
    random_state=42,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1
)

# Train the model
print("\nTraining model (this will take a minute)...")
fc_model.fit(X_train, y_train)

# Evaluate
accuracy = fc_model.score(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.2%}")

# Count parameters
params = 0
for i in range(len(fc_model.coefs_)):
    params += fc_model.coefs_[i].size
    params += fc_model.intercepts_[i].size

print(f"📊 Total Parameters: {params:,}")

# Save model
with open('mnist_fc_model.pkl', 'wb') as f:
    pickle.dump(fc_model, f)

print("💾 Model saved as 'mnist_fc_model.pkl'")

# Also save in H5 format for compatibility
try:
    # Try to save as H5 using a simple wrapper
    class ModelWrapper:
        def __init__(self, mlp_model):
            self.mlp_model = mlp_model
            self.input_shape = (784,)
            self.output_shape = (10,)
        
        def predict(self, X):
            return self.mlp_model.predict_proba(X)
        
        def count_params(self):
            params = 0
            for coef in self.mlp_model.coefs_:
                params += coef.size
            for intercept in self.mlp_model.intercepts_:
                params += intercept.size
            return params
    
    wrapped_model = ModelWrapper(fc_model)
    
    with open('mnist_fc_model_h5.pkl', 'wb') as f:
        pickle.dump(wrapped_model, f)
    
    print("💾 Model also saved as 'mnist_fc_model_h5.pkl' (H5 wrapper)")
    
except Exception as e:
    print(f"Note: Could not create H5 wrapper: {e}")

print("\n🎯 Training complete! Now update app.py to use both models.")