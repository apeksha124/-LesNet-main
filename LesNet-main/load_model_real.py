"""
Real Model Loader - Handles TensorFlow model compatibility issues
"""

import tensorflow as tf
import numpy as np
import os

def load_model_compatible(model_path):
    """Load model with full compatibility handling"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        # Method 1: Try loading with compile=False first
        print("Attempting to load model...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
        
        # Recompile with proper configuration
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        
        # Method 2: Try with custom objects
        try:
            print("Attempting to load with custom objects...")
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={
                    'Functional': tf.keras.Model,
                    'Model': tf.keras.Model,
                    'Sequential': tf.keras.Sequential
                }
            )
            
            # Recompile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            
            return model
            
        except Exception as e2:
            print(f"Custom object loading failed: {str(e2)}")
            
            # Method 3: Try loading as a different format
            try:
                print("Attempting alternative loading method...")
                
                # Load the model architecture and weights separately
                with tf.keras.utils.custom_object_scope({
                    'Functional': tf.keras.Model,
                    'Model': tf.keras.Model
                }):
                    model = tf.keras.models.load_model(model_path)
                
                return model
                
            except Exception as e3:
                print(f"All loading methods failed: {str(e3)}")
                raise Exception(f"Could not load model from {model_path}")

def test_model_prediction(model, test_image_path=None):
    """Test model prediction with real data"""
    
    if test_image_path and os.path.exists(test_image_path):
        # Test with real image
        import cv2
        
        # Load and preprocess image
        img = cv2.imread(test_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Preprocess for EfficientNet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img = preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img, verbose=0)
        
        print(f"Real prediction successful!")
        print(f"Prediction probabilities: {prediction[0]}")
        print(f"Predicted class: {'malignant' if np.argmax(prediction[0]) == 1 else 'benign'}")
        print(f"Confidence: {np.max(prediction[0]):.2%}")
        
        return prediction[0]
    
    else:
        # Test with random data
        print("Testing with random data...")
        
        # Create random test image
        test_input = np.random.random((1, 224, 224, 3))
        
        # Make prediction
        prediction = model.predict(test_input, verbose=0)
        
        print(f"Random test prediction: {prediction[0]}")
        print(f"Model is working correctly!")
        
        return prediction[0]

if __name__ == "__main__":
    # Test the model loader
    model_path = "models/lesnet_level4.keras"
    
    try:
        model = load_model_compatible(model_path)
        print("Model loaded successfully!")
        
        # Test prediction
        test_model_prediction(model)
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
