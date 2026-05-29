"""
ML Logic - Machine Learning and Ensemble System
Extracted from app_production_fixed.py for better organization
"""

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.utils import register_keras_serializable

# Custom BatchNormalization for TensorFlow compatibility
class CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
    """Custom BatchNormalization that handles incompatible parameters"""
    
    def __init__(self, *args, **kwargs):
        # Remove incompatible parameters
        incompatible_params = ['renorm', 'renorm_clipping', 'renorm_momentum']
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key not in incompatible_params:
                filtered_kwargs[key, value] = value
        super().__init__(*args, **filtered_kwargs)

# Register the custom layer
register_keras_serializable()(CompatibleBatchNormalization)

class MultiModelEnsemble:
    """Multi-model ensemble system for improved accuracy and reliability"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.model_metadata = {}
        self.ensemble_loaded = False
        self.primary_model = None
        self._load_model_ensemble()
    
    def _load_model_ensemble(self):
        """Load multiple models with intelligent weighting"""
        try:
            print("Loading multi-model ensemble...")
            
            # Define model configurations with weights based on performance
            model_configs = {
                'normal_skin_3class': {
                    'path': 'models/normal_skin_3class.keras',
                    'weight': 0.25,
                    'name': 'Normal Skin 3-Class Model',
                    'description': '3-class model (NORMAL SKIN, benign, malignant)'
                },
                'efficientnet_b0': {
                    'path': 'models/efficientnetb0.keras',
                    'weight': 0.25,
                    'name': 'EfficientNet-B0',
                    'description': 'EfficientNet-B0 pre-trained model'
                },
                'efficientnet_b3': {
                    'path': 'models/efficientnetb3.keras',
                    'weight': 0.25,
                    'name': 'EfficientNet-B3',
                    'description': 'EfficientNet-B3 pre-trained model'
                },
                'resnet50': {
                    'path': 'models/resnet50.keras',
                    'weight': 0.25,
                    'name': 'ResNet-50',
                    'description': 'ResNet-50 pre-trained model'
                }
            }
            
            # Load models with error handling
            loaded_count = 0
            for model_id, config in model_configs.items():
                try:
                    print(f"Loading model {model_id}: {config['name']}")
                    model = tf.keras.models.load_model(
                        config['path'], 
                        compile=False,
                        custom_objects={'BatchNormalization': CompatibleBatchNormalization}
                    )
                    self.models[model_id] = model
                    self.model_weights[model_id] = config['weight']
                    self.model_metadata[model_id] = {
                        'name': config['name'],
                        'description': config['description'],
                        'loaded': True
                    }
                    loaded_count += 1
                    print(f"Successfully loaded {model_id}")
                    
                except Exception as e:
                    print(f"Failed to load {model_id}: {str(e)}")
                    self.model_metadata[model_id] = {
                        'name': config['name'],
                        'description': config['description'],
                        'loaded': False,
                        'error': str(e)
                    }
            
            if loaded_count > 0:
                self.ensemble_loaded = True
                print(f"Loaded {loaded_count}/{len(model_configs)} models successfully")
            else:
                print("Failed to load any models")
                
        except Exception as e:
            print(f"Error loading ensemble: {str(e)}")
            self.ensemble_loaded = False
    
    def make_prediction(self, image_array):
        """Make prediction using loaded model"""
        if self.model and self.model_loaded:
            try:
                # Preprocess image for model
                model_input = self._preprocess_image(image_array)
                
                # Make prediction
                prediction = self.model.predict(model_input, verbose=0)
                pred_class_idx = np.argmax(prediction[0])
                confidence = float(prediction[0][pred_class_idx])
                
                return {
                    'pred_class': pred_class_idx,
                    'confidence': confidence,
                    'prediction': prediction[0],
                    'model_used': 'single_model'
                }
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                return None
        return None
    
    def _preprocess_image(self, image_array):
        """Preprocess image for model input"""
        try:
            # Ensure correct shape and type
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Convert to float32 if needed
            if image_array.dtype != np.float32:
                image_array = image_array.astype(np.float32)
            
            # Apply preprocessing
            processed = efficientnet_preprocess(image_array)
            return processed
            
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return None
    
    def predict_ensemble(self, image_array):
        """Make ensemble prediction using all loaded models"""
        if not self.ensemble_loaded:
            print("Ensemble not loaded")
            return None, 0.0, 'Error', {'error': 'Ensemble not loaded'}
        
        try:
            print("Using multi-model ensemble prediction")
            
            predictions = []
            confidences = []
            model_results = []
            
            # Get original image from preprocessed array (reverse preprocessing)
            original_image = np.clip(image_array[0] + [123.675, 116.28, 103.53], 0, 255).astype(np.uint8)
            
            for model_id, model in self.models.items():
                if model is not None:
                    try:
                        # Use model-specific preprocessing
                        model_input = self._get_model_specific_preprocessing(original_image, model_id)
                        
                        # Add batch dimension
                        model_input = np.expand_dims(model_input, axis=0)
                        
                        # Make prediction (fast - no verbose)
                        prediction = model.predict(model_input, verbose=0)
                        pred_class_idx = np.argmax(prediction[0])
                        confidence = float(prediction[0][pred_class_idx])
                        
                        predictions.append(pred_class_idx)
                        confidences.append(confidence)
                        
                        # Store individual model result
                        model_results.append({
                            'model_name': self.model_metadata[model_id]['name'],
                            'pred_class': pred_class_idx,
                            'confidence': confidence
                        })
                        
                        print(f"  {self.model_metadata[model_id]['name']}: {pred_class_idx} ({confidence:.3f})")
                    except Exception as e:
                        print(f"Error in {model_id}: {str(e)}")
                        continue
            
            if not predictions:
                return None, 0.0, 'Error', {'error': 'No valid predictions'}
            
            # Weighted ensemble prediction - convert to one-hot first
            num_classes = 3  # 3-class model: 0=NORMAL SKIN, 1=benign, 2=malignant
            ensemble_pred = np.zeros(num_classes)
            total_weight_used = 0
            
            for i, pred in enumerate(predictions):
                # Get weight from model weights dictionary
                model_id = list(self.models.keys())[i]
                weight = self.model_weights.get(model_id, 0.25)
                
                # Convert class index to one-hot vector
                one_hot = np.zeros(num_classes)
                one_hot[pred] = 1.0
                
                ensemble_pred += one_hot * weight
                total_weight_used += weight
            
            # Normalize if weights don't sum to 1
            if total_weight_used > 0:
                ensemble_pred /= total_weight_used
            
            # Calculate ensemble confidence and class
            ensemble_confidence = float(np.max(ensemble_pred))
            ensemble_class = int(np.argmax(ensemble_pred))
            
            # Calculate agreement metrics
            class_votes = [r['pred_class'] for r in model_results]
            agreement_rate = class_votes.count(ensemble_class) / len(class_votes)
            
            # Calculate confidence variance
            confidences = [r['confidence'] for r in model_results]
            confidence_variance = np.var(confidences)
            
            print(f"Ensemble result: {ensemble_class} ({ensemble_confidence:.3f})")
            print(f"Agreement: {agreement_rate:.2%}, Confidence variance: {confidence_variance:.4f}")
            
            return ensemble_pred, ensemble_confidence, {
                'class': ensemble_class,
                'model_results': model_results,
                'agreement_rate': agreement_rate,
                'confidence_variance': confidence_variance,
                'ensemble_size': len(predictions),
                'individual_predictions': {i: pred for i, pred in enumerate(predictions)},
                'original_image': image_array,
                'ensemble_used': True
            }
            
        except Exception as e:
            print(f"Ensemble prediction error: {str(e)}")
            return None, None, None
    
    def _get_model_specific_preprocessing(self, original_image, model_id):
        """Get model-specific preprocessing function"""
        preprocessing_map = {
            'normal_skin_3class': mobilenet_preprocess,
            'efficientnet_b0': efficientnet_preprocess,
            'efficientnet_b3': efficientnet_preprocess,
            'resnet50': resnet_preprocess
        }
        
        preprocess_func = preprocessing_map.get(model_id, efficientnet_preprocess)
        return preprocess_func(original_image.astype(np.float32))
    
    def get_model_status(self):
        """Get status of all models in the ensemble"""
        status = {
            'ensemble_loaded': self.ensemble_loaded,
            'total_models': len(self.model_metadata),
            'loaded_models': len(self.models),
            'models': {}
        }
        
        for model_id, metadata in self.model_metadata.items():
            status['models'][model_id] = {
                'name': metadata['name'],
                'description': metadata['description'],
                'loaded': metadata.get('loaded', False),
                'weight': self.model_weights.get(model_id, 0),
                'error': metadata.get('error', None)
            }
        
        return status

class GradCAMGenerator:
    """Medical-focused Grad-CAM with failure-proof design"""
    
    def __init__(self, model):
        self.model = model
        self.target_layers = []
        self.best_layer = None
        self.medical_focus_regions = self._define_medical_regions()
        self._setup_gradcam()
    
    def _define_medical_regions(self):
        """Define medically relevant attention regions for skin lesions"""
        return {
            'lesion': {'weight': 1.0, 'size': 0.15}
        }
    
    def _setup_gradcam(self):
        """Setup Grad-CAM with layer detection"""
        try:
            # Find the last convolutional layer
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    self.best_layer = layer
                    self.target_layers = [layer.name]
                    print(f"Found target layer: {layer.name}")
                    break
            
            if not self.best_layer:
                print("No suitable convolutional layer found")
                return False
            
            return True
            
        except Exception as e:
            print(f"Grad-CAM setup error: {str(e)}")
            return False
    
    def generate_gradcam(self, image_array, pred_class_idx):
        """Generate Grad-CAM heatmap with medical focus"""
        try:
            if not self.best_layer:
                print("Grad-CAM not properly setup")
                return None
            
            # Create model that outputs layer activations
            grad_model = tf.keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[self.best_layer.output, self.model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_array)
                loss = predictions[:, pred_class_idx]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Create heatmap
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            
            # Normalize and resize
            heatmap = heatmap / np.max(heatmap)
            heatmap = cv2.resize(heatmap, (image_array.shape[2], image_array.shape[1]))
            
            return heatmap
            
        except Exception as e:
            print(f"Grad-CAM generation error: {str(e)}")
            return None
    
    def apply_medical_focus(self, heatmap, pred_class_idx):
        """Apply medical focus weighting to heatmap"""
        try:
            if pred_class_idx in [1, 2]:  # benign or malignant
                # Enhance lesion regions
                h, w = heatmap.shape
                center_y, center_x = h // 2, w // 2
                radius = int(min(h, w) * self.medical_focus_regions['lesion']['size'])
                
                y, x = np.ogrid[:h, :w]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
                
                heatmap = heatmap * (1 + mask * self.medical_focus_regions['lesion']['weight'])
            
            return heatmap
            
        except Exception as e:
            print(f"Medical focus error: {str(e)}")
            return heatmap
    
    def create_overlay(self, original_image, heatmap, alpha=0.4):
        """Create Grad-CAM overlay with medical visualization"""
        try:
            # Convert heatmap to colormap
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap), 
                cv2.COLORMAP_JET
            )
            
            # Resize to match original image
            heatmap_colored = cv2.resize(heatmap_colored, 
                                       (original_image.shape[2], original_image.shape[1]))
            
            # Overlay heatmap on original image
            superimposed_img = cv2.addWeighted(original_image, 1-alpha, heatmap_colored, alpha, 0)
            
            return superimposed_img
        
        except Exception as e:
            print(f"Heatmap overlay error: {str(e)}")
            return original_image

def format_prediction_results(pred_class_idx, confidence, class_mapping):
    """Format prediction results for display"""
    return {
        'class_idx': pred_class_idx,
        'class_name': class_mapping.get(pred_class_idx, 'Unknown'),
        'confidence': confidence,
        'confidence_percent': f"{confidence:.1%}",
        'risk_level': get_risk_level(pred_class_idx, confidence),
        'recommendation': get_medical_recommendation(pred_class_idx, confidence)
    }

def get_risk_level(pred_class_idx, confidence):
    """Determine risk level based on prediction"""
    if pred_class_idx == 0:  # Normal skin
        return 'low'
    elif pred_class_idx == 1:  # Benign
        return confidence > 0.7 and 'medium' or 'low'
    else:  # Malignant
        return confidence > 0.8 and 'high' or 'medium'

def get_medical_recommendation(pred_class_idx, confidence):
    """Get medical recommendation based on prediction"""
    recommendations = {
        0: {  # Normal skin
            'risk_level': 'low',
            'risk_score': 0.1,
            'patient_education': [
                "Continue regular skin care routine",
                "Perform monthly self-examinations",
                "Use sunscreen daily",
                "Maintain healthy lifestyle"
            ],
            'care': [
                "Moisturize regularly",
                "Protect from sun exposure",
                "Monitor for changes"
            ]
        },
        1: {  # Benign
            'risk_level': 'low',
            'risk_score': 0.3,
            'patient_education': [
                "Monitor lesion for changes",
                "Document with photos monthly",
                "Consult dermatologist if changes occur",
                "Maintain regular check-ups"
            ],
            'care': [
                "Observe for growth",
                "Note color/texture changes",
                "Keep area clean and dry"
            ]
        },
        2: {  # Malignant
            'risk_level': 'high',
            'risk_score': 0.8,
            'patient_education': [
                "IMMEDIATE medical consultation required",
                "Do not self-diagnose or delay treatment",
                "Follow dermatologist recommendations",
                "Consider second opinion for confirmation"
            ],
            'care': [
                "Seek immediate medical attention",
                "Follow treatment plan strictly",
                "Document treatment progress"
            ]
        }
    }
    
    return recommendations.get(pred_class_idx, recommendations[0])

def calculate_ensemble_metrics(model_results):
    """Calculate ensemble performance metrics"""
    if not model_results:
        return {}
    
    confidences = [r['confidence'] for r in model_results]
    predictions = [r['pred_class'] for r in model_results]
    
    # Find most common prediction
    from collections import Counter
    most_common = Counter(predictions).most_common(1)[0]
    
    return {
        'agreement_rate': predictions.count(most_common[0]) / len(predictions),
        'confidence_variance': np.var(confidences),
        'confidence_mean': np.mean(confidences),
        'confidence_std': np.std(confidences),
        'consensus_class': most_common[0],
        'consensus_strength': most_common[1] / len(predictions)
    }

def validate_prediction_confidence(confidence, threshold=0.5):
    """Validate if confidence meets minimum threshold"""
    return confidence >= threshold

def apply_medical_confidence_calibration(confidence, pred_class_idx):
    """Apply medical safety calibration to confidence"""
    # Medical systems should never claim 100% certainty
    if confidence > 0.95:
        # Apply temperature scaling
        temperature = 2.0
        calibrated_confidence = confidence / temperature
        return min(calibrated_confidence, 0.98)  # Max 98% for medical safety
    
    return confidence

def get_ensemble_weight_distribution(model_weights):
    """Get normalized weight distribution"""
    total_weight = sum(model_weights.values())
    return {k: v/total_weight for k, v in model_weights.items()}

def detect_prediction_anomalies(model_results):
    """Detect anomalies in model predictions"""
    if not model_results:
        return []
    
    anomalies = []
    confidences = [r['confidence'] for r in model_results]
    
    # Check for extreme confidence differences
    max_conf = max(confidences)
    min_conf = min(confidences)
    
    if max_conf - min_conf > 0.5:
        anomalies.append("Large confidence variance between models")
    
    # Check for low overall confidence
    avg_confidence = np.mean(confidences)
    if avg_confidence < 0.6:
        anomalies.append("Low overall confidence across models")
    
    return anomalies
