"""
3-Label Model Training - NORMAL SKIN, LICHEN PLANUS, OTHER CONDITIONS
Preserves original working script while adding 3-label support
"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import json
from datetime import datetime

def create_3label_model():
    """Create EfficientNetB0 model for 3-label classification"""
    
    print("Creating EfficientNetB0 model for 3-label classification...")
    
    # Create base model
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base layers initially
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"Model created with {model.count_params():,} parameters")
    return model, base_model

def train_3label_model():
    """Train the 3-label model with organized dataset"""
    
    print("SkinVestigator AI - 3-Label Model Training")
    print("=" * 60)
    print("Labels: NORMAL SKIN, LICHEN PLANUS, OTHER CONDITIONS")
    
    # Check dataset
    data_dir = "Dataset/train"
    if not os.path.exists(data_dir):
        print(f"Error: Dataset not found at {data_dir}")
        return None
    
    # Check for required folders
    required_folders = ["NORMAL SKIN", "LICHEN PLANUS"]
    for folder in required_folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Error: Required folder not found: {folder_path}")
            return None
    
    # Create data generators
    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=preprocess_input,
        data_format=None,
        validation_split=0.2,
        interpolation_order=1,
        dtype=None
    )
    
    val_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=preprocess_input,
        data_format=None,
        validation_split=0.2,
        interpolation_order=1,
        dtype=None
    )
    
    # Create generators
    train_data = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_data = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"Training samples: {len(train_data) * 32}")
    print(f"Validation samples: {len(val_data) * 32}")
    print(f"Number of classes: {train_data.num_classes}")
    print(f"Class names: {train_data.class_indices}")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_data.classes),
        y=train_data.classes
    )
    class_weights = dict(enumerate(class_weights))
    
    print(f"Class weights: {class_weights}")
    
    # Create model
    model, base_model = create_3label_model()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        min_delta=0.01,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Phase 1: Train with frozen base layers
    print("\nPhase 1: Training with frozen base layers...")
    history_phase1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=15,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\nPhase 2: Fine-tuning top layers...")
    
    # Unfreeze top layers
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Get the total number of layers
    total_layers = len(base_model.layers)
    print(f"Base model has {total_layers} layers")
    
    # Freeze bottom layers, unfreeze top 50
    for i, layer in enumerate(base_model.layers):
        if i < (total_layers - 50):
            layer.trainable = False
        else:
            layer.trainable = True
    
    print(f"Unfroze top 50 layers of base model")
    
    # Re-compile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    history_phase2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Save model
    model_path = "models/3label_efficientnet.keras"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate model
    print("\nEvaluating model...")
    val_results = model.evaluate(val_data, verbose=1)
    print(f"Validation Loss: {val_results[0]:.4f}")
    print(f"Validation Accuracy: {val_results[1]:.4f}")
    print(f"Validation AUC: {val_results[2]:.4f}")
    
    # Test prediction
    print("\nTesting prediction...")
    test_batch = next(iter(val_data))
    test_images, test_labels = test_batch
    predictions = model.predict(test_images[:1], verbose=0)
    
    print(f"Test prediction: {predictions[0]}")
    print(f"Predicted class: {np.argmax(predictions[0])}")
    print(f"True class: {np.argmax(test_labels[0])}")
    print(f"Confidence: {np.max(predictions[0]):.2%}")
    
    # Save training info
    training_info = {
        'model_path': model_path,
        'training_date': datetime.now().isoformat(),
        'validation_accuracy': float(val_results[1]),
        'validation_auc': float(val_results[2]),
        'class_weights': class_weights,
        'class_indices': train_data.class_indices,
        'phase1_best_epoch': early_stop.best_epoch,
        'total_layers': total_layers,
        'model_type': '3label_classification'
    }
    
    with open("models/3label_training_info.json", 'w') as f:
        json.dump(training_info, f, indent=4)
    
    print(f"\n3-Label training completed successfully!")
    print(f"Model can now detect NORMAL SKIN vs conditions!")
    print(f"Use model path: {model_path}")
    
    return model, model_path

if __name__ == "__main__":
    model, model_path = train_3label_model()
    
    if model:
        print(f"\n" + "="*60)
        print("SUCCESS: 3-Label model trained and saved!")
        print("Ready for normal skin detection!")
        print("="*60)
    else:
        print("\nFAILED: Could not train 3-label model")
