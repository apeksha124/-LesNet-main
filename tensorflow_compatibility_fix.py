import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import register_keras_serializable

class CompatibleBatchNormalization(BatchNormalization):
    """Custom BatchNormalization that handles incompatible parameters"""
    
    def __init__(self, *args, **kwargs):
        # Remove incompatible parameters that newer TensorFlow models might have
        incompatible_params = ['renorm', 'renorm_clipping', 'renorm_momentum']
        
        # Filter out incompatible parameters
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key not in incompatible_params:
                filtered_kwargs[key] = value
        
        # Initialize parent with filtered parameters
        super().__init__(*args, **filtered_kwargs)
        
        print(f"CompatibleBatchNormalization: Filtered out incompatible params: {[k for k in incompatible_params if k in kwargs]}")

# Register the custom layer
register_keras_serializable()(CompatibleBatchNormalization)

def test_compatibility():
    """Test if 3-class models load with compatibility fix"""
    try:
        # Test loading 3-class model with custom object
        model = tf.keras.models.load_model(
            'models/normal_skin_3class.keras',
            custom_objects={'BatchNormalization': CompatibleBatchNormalization},
            compile=False
        )
        print("✅ Successfully loaded 3-class model with compatibility fix!")
        
        # Test prediction
        import numpy as np
        dummy_input = np.random.random((1, 224, 224, 3))
        pred = model.predict(dummy_input, verbose=0)
        print(f"Prediction shape: {pred.shape}")
        print(f"Classes: {len(pred[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load 3-class model: {e}")
        return False

if __name__ == "__main__":
    success = test_compatibility()
    if success:
        print("\n🎉 TensorFlow compatibility issue RESOLVED!")
        print("Now you can use 3-class models in the app.")
    else:
        print("\n❌ TensorFlow compatibility issue NOT resolved.")
