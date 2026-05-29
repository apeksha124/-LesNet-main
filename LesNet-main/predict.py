import numpy as np
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

logger = logging.getLogger(__name__)

# Load model once
model = load_model("models/skinvestigator_nano_40MB_91_38_acc.h5", compile=False)

# Class labels - these match the 8 classes the model was actually trained on
labels = [
    "akiec", "bcc", "bkl", "df",
    "melanoma", "nevus", "vasc", "non_lesion"
]

def predict(model, img_array, labels):
    """Simple prediction - always return the top prediction"""
    try:
        # Get predictions
        preds = model.predict(img_array, verbose=0)[0]
        
        # Get top prediction and confidence
        top_idx = np.argmax(preds)
        top_confidence = float(preds[top_idx])
        
        # Always return the top prediction - no validation blocking
        logger.info(f"Prediction: {labels[top_idx]}, Confidence: {top_confidence:.3f}")
        
        return labels[top_idx], top_confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "error", 0.0

def get_all_predictions(model, img_array, labels, top_k=3):
    """Get top k predictions with details"""
    try:
        preds = model.predict(img_array, verbose=0)[0]
        
        # Get top k indices and predictions
        top_indices = np.argsort(preds)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'label': labels[idx],
                'confidence': float(preds[idx]),
                'index': int(idx)
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting all predictions: {str(e)}")
        return []