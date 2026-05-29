"""
3-Label Prediction Script - Handles NORMAL SKIN detection
Shows "No lesion detected" for normal skin predictions
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import json
import os

class SkinLesionPredictor3Label:
    def __init__(self, model_path="models/3label_efficientnet.keras"):
        """Initialize the 3-label predictor"""
        self.model_path = model_path
        self.model = None
        self.class_indices = None
        self.class_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained 3-label model"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Load training info to get class indices
            info_path = "models/3label_training_info.json"
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    training_info = json.load(f)
                    self.class_indices = training_info['class_indices']
                    # Create reverse mapping
                    self.class_names = {v: k for k, v in self.class_indices.items()}
                    print(f"Loaded class mapping: {self.class_names}")
            else:
                print("Warning: Training info not found, using default mapping")
                self.class_names = {0: "NORMAL SKIN", 1: "LICHEN PLANUS", 2: "OTHER CONDITIONS"}
            
            print(f"3-Label model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, img_path):
        """Preprocess image for prediction"""
        try:
            # Load and preprocess image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, img_path):
        """Make prediction on image"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Preprocess image
        processed_img = self.preprocess_image(img_path)
        if processed_img is None:
            return {"error": "Failed to preprocess image"}
        
        try:
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Get class name
            predicted_class = self.class_names.get(predicted_class_idx, "UNKNOWN")
            
            # Special handling for normal skin
            if predicted_class == "NORMAL SKIN":
                result = {
                    "prediction": "No lesion detected",
                    "condition": "Normal Skin",
                    "confidence": f"{confidence:.2%}",
                    "is_normal": True,
                    "all_probabilities": {
                        "normal_skin": f"{predictions[0][list(self.class_names.keys())[list(self.class_names.values()).index('NORMAL SKIN')]]:.2%}" if 'NORMAL SKIN' in self.class_names.values() else "0%",
                        "lichen_planus": f"{predictions[0][list(self.class_names.keys())[list(self.class_names.values()).index('LICHEN PLANUS')]]:.2%}" if 'LICHEN PLANUS' in self.class_names.values() else "0%",
                        "other_conditions": f"{predictions[0][list(self.class_names.keys())[list(self.class_names.values()).index('OTHER CONDITIONS')]]:.2%}" if 'OTHER CONDITIONS' in self.class_names.values() else "0%"
                    }
                }
            else:
                result = {
                    "prediction": predicted_class,
                    "condition": predicted_class,
                    "confidence": f"{confidence:.2%}",
                    "is_normal": False,
                    "all_probabilities": {
                        "normal_skin": f"{predictions[0][list(self.class_names.keys())[list(self.class_names.values()).index('NORMAL SKIN')]]:.2%}" if 'NORMAL SKIN' in self.class_names.values() else "0%",
                        "lichen_planus": f"{predictions[0][list(self.class_names.keys())[list(self.class_names.values()).index('LICHEN PLANUS')]]:.2%}" if 'LICHEN PLANUS' in self.class_names.values() else "0%",
                        "other_conditions": f"{predictions[0][list(self.class_names.keys())[list(self.class_names.values()).index('OTHER CONDITIONS')]]:.2%}" if 'OTHER CONDITIONS' in self.class_names.values() else "0%"
                    }
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

# Test function
def test_prediction():
    """Test the 3-label prediction system"""
    print("Testing 3-Label Prediction System")
    print("=" * 50)
    
    predictor = SkinLesionPredictor3Label()
    
    # Test with a sample image if available
    test_image = "Dataset/train/NORMAL SKIN/NS_00001.jpg"
    if os.path.exists(test_image):
        print(f"Testing with: {test_image}")
        result = predictor.predict(test_image)
        print(f"Result: {result}")
    else:
        print("Test image not found, but predictor is ready")
    
    return predictor

if __name__ == "__main__":
    predictor = test_prediction()
