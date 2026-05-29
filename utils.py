"""
Utils - Helper Functions and Constants
Extracted from app_production_fixed.py for better organization
"""

import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# CLASS MAPPINGS
CLASS_MAPPING = {
    0: 'NORMAL SKIN',
    1: 'benign', 
    2: 'malignant'
}

CLASS_DESCRIPTIONS = {
    0: 'No lesion detected - Normal skin',
    1: 'Benign lesion - Non-cancerous growth',
    2: 'Malignant lesion - Cancerous growth requiring medical attention'
}

# MODEL CONFIGURATIONS
MODEL_CONFIGS = {
    'normal_skin_3class': {
        'path': 'models/normal_skin_3class.keras',
        'weight': 0.25,
        'name': 'Normal Skin 3-Class Model',
        'description': '3-class model (NORMAL SKIN, benign, malignant)',
        'preprocessing': 'mobilenet_preprocess'
    },
    'efficientnet_b0': {
        'path': 'models/efficientnetb0_new.keras',
        'weight': 0.25,
        'name': 'EfficientNet-B0',
        'description': 'EfficientNet-B0 pre-trained model',
        'preprocessing': 'efficientnet_preprocess'
    },
    'efficientnet_b3': {
        'path': 'models/efficientnetb3_new.keras',
        'weight': 0.25,
        'name': 'EfficientNet-B3',
        'description': 'EfficientNet-B3 pre-trained model',
        'preprocessing': 'efficientnet_preprocess'
    },
    'resnet50': {
        'path': 'models/resnet50_new.keras',
        'weight': 0.25,
        'name': 'ResNet-50',
        'description': 'ResNet-50 pre-trained model',
        'preprocessing': 'resnet_preprocess'
    }
}

# IMAGE CONFIGURATIONS
DEFAULT_IMAGE_SIZE = (224, 224)
EFFICIENTNET_B3_SIZE = (300, 300)
BATCH_SIZE = 32

# CONFIDENCE THRESHOLDS
CONFIDENCE_THRESHOLDS = {
    'low': 0.5,
    'medium': 0.7,
    'high': 0.8,
    'medical_max': 0.98  # Maximum 98% for medical safety
}

# RISK LEVELS
RISK_LEVELS = {
    'low': {
        'color': 'green',
        'description': 'Low risk - Regular monitoring recommended',
        'score_range': (0.0, 0.4)
    },
    'medium': {
        'color': 'orange', 
        'description': 'Medium risk - Medical consultation recommended',
        'score_range': (0.4, 0.7)
    },
    'high': {
        'color': 'red',
        'description': 'High risk - Immediate medical attention required',
        'score_range': (0.7, 1.0)
    }
}

# PREPROCESSING FUNCTIONS
def get_preprocessing_function(model_id):
    """Get appropriate preprocessing function for model"""
    return MODEL_CONFIGS.get(model_id, {}).get('preprocessing', efficientnet_preprocess)

def preprocess_image_for_model(image, model_id):
    """Apply model-specific preprocessing to image"""
    try:
        preprocess_func = get_preprocessing_function(model_id)
        
        # Ensure image is in correct format
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Apply model-specific preprocessing
        processed = preprocess_func(image)
        return processed
        
    except Exception as e:
        print(f"Preprocessing error for {model_id}: {str(e)}")
        return None

def reverse_preprocessing(image_array):
    """Reverse preprocessing to get original image"""
    try:
        # Add ImageNet mean values back
        original_image = np.clip(image_array[0] + [123.675, 116.28, 103.53], 0, 255).astype(np.uint8)
        return original_image
    except Exception as e:
        print(f"Reverse preprocessing error: {str(e)}")
        return None

# HELPER FUNCTIONS
def format_confidence(confidence):
    """Format confidence as percentage"""
    return f"{confidence:.1%}"

def format_confidence_decimal(confidence):
    """Format confidence as decimal"""
    return f"{confidence:.3f}"

def get_class_name(class_idx):
    """Get class name from index"""
    return CLASS_MAPPING.get(class_idx, 'Unknown')

def get_class_description(class_idx):
    """Get class description from index"""
    return CLASS_DESCRIPTIONS.get(class_idx, 'Unknown class')

def get_risk_level(pred_class_idx, confidence):
    """Determine risk level based on prediction"""
    if pred_class_idx == 0:  # Normal skin
        return 'low'
    elif pred_class_idx == 1:  # Benign
        return confidence > CONFIDENCE_THRESHOLDS['medium'] and 'medium' or 'low'
    else:  # Malignant
        return confidence > CONFIDENCE_THRESHOLDS['high'] and 'high' or 'medium'

def get_risk_color(risk_level):
    """Get color for risk level"""
    return RISK_LEVELS.get(risk_level, {}).get('color', 'gray')

def get_risk_description(risk_level):
    """Get description for risk level"""
    return RISK_LEVELS.get(risk_level, {}).get('description', 'Unknown risk level')

def get_model_weight(model_id):
    """Get model weight from configuration"""
    return MODEL_CONFIGS.get(model_id, {}).get('weight', 0.25)

def get_model_path(model_id):
    """Get model path from configuration"""
    return MODEL_CONFIGS.get(model_id, {}).get('path', '')

def get_model_name(model_id):
    """Get model name from configuration"""
    return MODEL_CONFIGS.get(model_id, {}).get('name', 'Unknown Model')

def validate_confidence(confidence, threshold='medium'):
    """Validate if confidence meets threshold"""
    min_confidence = CONFIDENCE_THRESHOLDS.get(threshold, 0.5)
    return confidence >= min_confidence

def apply_medical_confidence_calibration(confidence, pred_class_idx):
    """Apply medical safety calibration to confidence"""
    # Medical systems should never claim 100% certainty
    if confidence > CONFIDENCE_THRESHOLDS['medical_max']:
        # Apply temperature scaling
        temperature = 2.0
        calibrated_confidence = confidence / temperature
        return min(calibrated_confidence, CONFIDENCE_THRESHOLDS['medical_max'])
    
    return confidence

def calculate_risk_score(pred_class_idx, confidence):
    """Calculate risk score based on prediction"""
    base_scores = {0: 0.1, 1: 0.3, 2: 0.8}
    confidence_factor = min(confidence, 1.0)
    return base_scores.get(pred_class_idx, 0.5) * confidence_factor

def format_prediction_result(pred_class_idx, confidence):
    """Format prediction result for display"""
    return {
        'class_idx': pred_class_idx,
        'class_name': get_class_name(pred_class_idx),
        'class_description': get_class_description(pred_class_idx),
        'confidence': confidence,
        'confidence_percent': format_confidence(confidence),
        'risk_level': get_risk_level(pred_class_idx, confidence),
        'risk_color': get_risk_color(get_risk_level(pred_class_idx, confidence)),
        'risk_score': calculate_risk_score(pred_class_idx, confidence)
    }

def format_model_result(model_name, pred_class_idx, confidence):
    """Format individual model result"""
    return {
        'model_name': model_name,
        'pred_class': pred_class_idx,
        'class_name': get_class_name(pred_class_idx),
        'confidence': confidence,
        'confidence_percent': format_confidence(confidence)
    }

def validate_image_array(image_array):
    """Validate image array format"""
    if image_array is None:
        return False, "Image array is None"
    
    if not isinstance(image_array, np.ndarray):
        return False, "Image is not a numpy array"
    
    if len(image_array.shape) not in [3, 4]:
        return False, f"Invalid image shape: {image_array.shape}"
    
    if image_array.dtype not in [np.uint8, np.float32, np.float64]:
        return False, f"Invalid image dtype: {image_array.dtype}"
    
    return True, "Valid image array"

def resize_image(image, target_size):
    """Resize image to target size"""
    try:
        if len(image.shape) == 3:
            return cv2.resize(image, target_size)
        elif len(image.shape) == 4:
            return cv2.resize(image, target_size[:2])
        else:
            return None
    except Exception as e:
        print(f"Resize error: {str(e)}")
        return None

def normalize_image(image):
    """Normalize image values"""
    try:
        return image.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Normalization error: {str(e)}")
        return None

def denormalize_image(image):
    """Denormalize image values"""
    try:
        return (image * 255.0).astype(np.uint8)
    except Exception as e:
        print(f"Denormalization error: {str(e)}")
        return None

# DEBUG FUNCTIONS
def debug_print(message, level="INFO"):
    """Print debug message with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def debug_model_prediction(model_id, prediction, confidence):
    """Debug model prediction"""
    debug_print(f"{model_id}: pred={np.argmax(prediction)} conf={confidence:.3f}")

def debug_ensemble_result(ensemble_class, ensemble_confidence, agreement_rate):
    """Debug ensemble result"""
    debug_print(f"Ensemble: class={ensemble_class} conf={ensemble_confidence:.3f} agreement={agreement_rate:.1%}")

# FILE OPERATIONS
def ensure_directory_exists(directory_path):
    """Ensure directory exists"""
    import os
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        debug_print(f"Created directory: {directory_path}")

def get_file_size(file_path):
    """Get file size in MB"""
    import os
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    return 0

def validate_model_file(model_path):
    """Validate model file exists and is readable"""
    import os
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"
    
    if not model_path.endswith('.keras') and not model_path.endswith('.h5'):
        return False, f"Invalid model file format: {model_path}"
    
    return True, "Model file is valid"

# PERFORMANCE METRICS
def calculate_accuracy_metrics(predictions, ground_truth):
    """Calculate accuracy metrics"""
    if not predictions or not ground_truth:
        return {}
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    total = len(predictions)
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct_predictions': correct,
        'total_predictions': total,
        'error_rate': (total - correct) / total if total > 0 else 0
    }

def calculate_confidence_statistics(confidences):
    """Calculate confidence statistics"""
    if not confidences:
        return {}
    
    return {
        'mean': np.mean(confidences),
        'std': np.std(confidences),
        'min': np.min(confidences),
        'max': np.max(confidences),
        'variance': np.var(confidences),
        'median': np.median(confidences)
    }

def calculate_ensemble_agreement(predictions):
    """Calculate ensemble agreement metrics"""
    if not predictions:
        return {}
    
    from collections import Counter
    prediction_counts = Counter(predictions)
    most_common = prediction_counts.most_common(1)[0]
    
    return {
        'agreement_rate': most_common[1] / len(predictions),
        'consensus_class': most_common[0],
        'consensus_strength': most_common[1] / len(predictions),
        'disagreement_rate': 1 - (most_common[1] / len(predictions))
    }
