"""
Normal Skin Prediction Script - Handles NORMAL SKIN + benign + malignant
Shows "No lesion detected" for normal skin predictions
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import json
import os

class NormalSkinPredictor:
    def __init__(self, model_path="models/normal_skin_3class.keras"):
        """Initialize the normal skin predictor"""
        self.model_path = model_path
        self.model = None
        self.class_indices = None
        self.class_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained normal skin model"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Load training info to get class indices
            info_path = "models/normal_skin_training_info.json"
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    training_info = json.load(f)
                    self.class_indices = training_info['class_indices']
                    # Create reverse mapping
                    self.class_names = {v: k for k, v in self.class_indices.items()}
                    print(f"Loaded class mapping: {self.class_names}")
            else:
                print("Warning: Training info not found, using default mapping")
                self.class_names = {0: "NORMAL SKIN", 1: "benign", 2: "malignant"}
            
            print(f"Normal Skin model loaded successfully from {self.model_path}")
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
                    "risk_level": "No Risk",
                    "recommendation": "No medical intervention needed",
                    "all_probabilities": self._format_probabilities(predictions[0])
                }
            elif predicted_class == "benign":
                result = {
                    "prediction": "Benign lesion detected",
                    "condition": "Benign",
                    "confidence": f"{confidence:.2%}",
                    "is_normal": False,
                    "risk_level": "Low Risk",
                    "recommendation": "Consult dermatologist for evaluation",
                    "all_probabilities": self._format_probabilities(predictions[0])
                }
            elif predicted_class == "malignant":
                result = {
                    "prediction": "Malignant lesion detected",
                    "condition": "Malignant",
                    "confidence": f"{confidence:.2%}",
                    "is_normal": False,
                    "risk_level": "High Risk",
                    "recommendation": "Immediate medical consultation required",
                    "all_probabilities": self._format_probabilities(predictions[0])
                }
            else:
                result = {
                    "prediction": "Unknown condition",
                    "condition": predicted_class,
                    "confidence": f"{confidence:.2%}",
                    "is_normal": False,
                    "risk_level": "Unknown",
                    "recommendation": "Consult dermatologist",
                    "all_probabilities": self._format_probabilities(predictions[0])
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def _format_probabilities(self, predictions):
        """Format all probabilities for output"""
        prob_dict = {}
        for class_idx, prob in enumerate(predictions):
            class_name = self.class_names.get(class_idx, f"Class_{class_idx}")
            prob_dict[class_name.lower().replace(" ", "_")] = f"{prob:.2%}"
        return prob_dict

# Test function
def test_normal_skin_prediction():
    """Test the normal skin prediction system"""
    print("Testing Normal Skin Prediction System")
    print("=" * 50)
    
    predictor = NormalSkinPredictor()
    
    # Test with a normal skin sample if available
    test_image = "Dataset/train/NORMAL SKIN/NS_00001.jpg"
    if os.path.exists(test_image):
        print(f"Testing with: {test_image}")
        result = predictor.predict(test_image)
        print(f"Result: {result}")
    else:
        print("Normal skin test image not found, but predictor is ready")
    
    # Test with benign sample if available
    benign_test = "Dataset/train/benign"
    if os.path.exists(benign_test):
        benign_files = [f for f in os.listdir(benign_test) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if benign_files:
            benign_test_image = os.path.join(benign_test, benign_files[0])
            print(f"\nTesting with benign: {benign_test_image}")
            result = predictor.predict(benign_test_image)
            print(f"Result: {result}")
    
    return predictor

if __name__ == "__main__":
    predictor = test_normal_skin_prediction()
