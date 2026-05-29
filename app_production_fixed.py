"""
Production Skin Vestigator App - Separated Version
Main application file with imports from separated components
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import json

# Import from separated files
from ui_components import (
    render_patient_education,
    render_medical_disclaimer,
    render_history_page,
    render_care_recommendations,
    render_risk_assessment,
    render_prediction_summary,
    render_gradcam_display,
    render_image_display,
    render_confidence_bar,
    render_class_probabilities,
    render_model_comparison,
    render_error_message,
    render_loading_indicator,
    render_success_message,
    render_info_message,
    render_warning_message,
    render_ensemble_info,
    render_consultation_summary,
    render_css_styles
)

from ml_logic import (
    MultiModelEnsemble,
    GradCAMGenerator,
    format_prediction_results,
    get_risk_level,
    get_medical_recommendation,
    calculate_ensemble_metrics,
    validate_prediction_confidence,
    apply_medical_confidence_calibration,
    get_ensemble_weight_distribution,
    detect_prediction_anomalies
)

from utils import (
    CLASS_MAPPING,
    CLASS_DESCRIPTIONS,
    MODEL_CONFIGS,
    DEFAULT_IMAGE_SIZE,
    BATCH_SIZE,
    CONFIDENCE_THRESHOLDS,
    RISK_LEVELS,
    get_preprocessing_function,
    preprocess_image_for_model,
    reverse_preprocessing,
    format_confidence,
    format_confidence_decimal,
    get_class_name,
    get_class_description,
    get_risk_color,
    get_risk_description,
    get_model_weight,
    get_model_path,
    get_model_name,
    validate_confidence,
    calculate_risk_score,
    format_prediction_result,
    format_model_result,
    validate_image_array,
    resize_image,
    normalize_image,
    denormalize_image,
    debug_print,
    debug_model_prediction,
    debug_ensemble_result,
    ensure_directory_exists,
    get_file_size,
    validate_model_file,
    calculate_accuracy_metrics,
    calculate_confidence_statistics,
    calculate_ensemble_agreement
)

class ProductionSkinVestigatorApp:
    """Production-ready healthcare AI with multi-model ensemble system"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.training_info = None
        self.gradcam = None
        self.model_ensemble = None
        self.use_ensemble = True  # Enable multi-model by default
        
        self._init_session_state()
        self._load_production_model()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'current_prediction' not in st.session_state:
            st.session_state.current_prediction = None
        if 'current_image_array' not in st.session_state:
            st.session_state.current_image_array = None
        if 'current_original_image' not in st.session_state:
            st.session_state.current_original_image = None
        if 'camera_image' not in st.session_state:
            st.session_state.camera_image = None
        if 'patient_records' not in st.session_state:
            st.session_state.patient_records = {}
        if 'consultations' not in st.session_state:
            st.session_state.consultations = []
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = False
        if 'skin_app' not in st.session_state:
            st.session_state.skin_app = self
    
    def _load_production_model(self):
        """Load production model(s)"""
        try:
            if self.use_ensemble:
                # Load multi-model ensemble
                self.model_ensemble = MultiModelEnsemble()
                if self.model_ensemble.ensemble_loaded:
                    print("Multi-model ensemble loaded successfully")
                else:
                    print("Failed to load ensemble, falling back to single model")
                    self.use_ensemble = False
            else:
                # Load single model (fallback)
                print("Loading single model...")
                # This would load a single model as fallback
                pass
            
        except Exception as e:
            print(f"Error loading production model: {str(e)}")
            self.model_loaded = False
    
    def preprocess_image_for_model(self, image, model_id):
        """Preprocess image for specific model"""
        return preprocess_image_for_model(image, model_id)
    
    def make_prediction(self, image_array):
        """Make prediction using ensemble or single model"""
        try:
            if self.use_ensemble and self.model_ensemble:
                return self.model_ensemble.predict_ensemble(image_array)
            else:
                # Fallback to single model prediction
                if self.model and self.model_loaded:
                    return self.model.make_prediction(image_array)
                else:
                    return None, 0.0, 'Error', {'error': 'No model loaded'}
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None, 0.0, 'Error', {'error': str(e)}
    
    def generate_gradcam_visualization(self, image_array, original_image, pred_class_idx):
        """Generate Grad-CAM visualization"""
        try:
            if self.use_ensemble and self.model_ensemble:
                # Use ensemble's primary model for Grad-CAM
                if hasattr(self.model_ensemble, 'primary_model') and self.model_ensemble.primary_model:
                    gradcam_gen = GradCAMGenerator(self.model_ensemble.primary_model)
                    heatmap = gradcam_gen.generate_gradcam(image_array, pred_class_idx)
                    
                    if heatmap is not None:
                        overlay = gradcam_gen.create_overlay(
                            np.clip(original_image[0] + [123.675, 116.28, 103.53], 0, 255).astype(np.uint8),
                            heatmap
                        )
                        return overlay
                    else:
                        return None
                else:
                    print("No primary model available for Grad-CAM")
                    return None
            else:
                # Single model Grad-CAM
                if self.model and self.model_loaded:
                    gradcam_gen = GradCAMGenerator(self.model)
                    heatmap = gradcam_gen.generate_gradcam(image_array, pred_class_idx)
                    
                    if heatmap is not None:
                        overlay = gradcam_gen.create_overlay(
                            np.clip(original_image[0] + [123.675, 116.28, 103.53], 0, 255).astype(np.uint8),
                            heatmap
                        )
                        return overlay
                    else:
                        return None
                else:
                    print("No model available for Grad-CAM")
                    return None
        except Exception as e:
            print(f"Grad-CAM generation error: {str(e)}")
            return None
    
    def perform_simple_analysis(self, image_array):
        """Perform analysis and display results"""
        try:
            # Clear previous results
            st.session_state.current_prediction = None
            
            # Make prediction
            ensemble_pred, ensemble_confidence, pred_details = self.make_prediction(image_array)
            
            if ensemble_pred is None:
                render_error_message("Failed to make prediction")
                return
            
            # Format prediction results
            pred_class_idx = pred_details['class']
            pred_result = format_prediction_results(pred_class_idx, ensemble_confidence)
            
            # Generate Grad-CAM
            original_image = reverse_preprocessing(image_array)
            gradcam_img = self.generate_gradcam_visualization(image_array, original_image, pred_class_idx)
            
            # Store prediction in session state
            st.session_state.current_prediction = {
                'prediction': ensemble_pred,
                'confidence': ensemble_confidence,
                'class': pred_class_idx,
                'class_name': pred_result['class_name'],
                'recommendation': pred_result['recommendation'],
                'risk_level': pred_result['recommendation']['risk_level'],
                'risk_score': pred_result['recommendation']['risk_score'],
                'gradcam_available': gradcam_img is not None,
                'image_source': st.session_state.get('image_source', 'unknown'),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_results': pred_details.get('model_results', []),
                'agreement_rate': pred_details.get('agreement_rate', 0),
                'individual_predictions': pred_details.get('individual_predictions', {}),
                'ensemble_used': pred_details.get('ensemble_used', False)
            }
            
            # Display results
            self._display_production_results()
            
        except Exception as e:
            render_error_message(f"Analysis error: {str(e)}")
    
    def _display_production_results(self):
        """Display production results with Grad-CAM"""
        if not st.session_state.current_prediction:
            return
        
        pred = st.session_state.current_prediction
        risk_level = pred['recommendation']['risk_level']
        image_source = pred.get('image_source', 'unknown')
        
        # Apply custom CSS
        render_css_styles()
        
        # Main results container
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        
        # Prediction summary
        render_prediction_summary(pred)
        
        # Risk assessment
        render_risk_assessment(pred)
        
        # Individual model results
        if pred.get('model_results'):
            render_model_comparison(pred['model_results'])
        
        # Ensemble information
        if pred.get('ensemble_used'):
            render_ensemble_info(pred)
        
        # Grad-CAM visualization
        if pred.get('gradcam_available', False):
            if st.session_state.get('current_original_image') is not None:
                render_gradcam_display(
                    st.session_state.current_original_image, 
                    pred['confidence']
                )
        
        # Care recommendations
        render_care_recommendations(pred)
        
        # Patient education
        render_patient_education(pred)
        
        # Medical disclaimer
        render_medical_disclaimer()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def preprocess_image_from_file(self, uploaded_file):
        """Preprocess image from uploaded file"""
        try:
            # Clear previous results
            st.session_state.current_prediction = None
            st.session_state.current_image_array = None
            st.session_state.current_original_image = None
            
            # Read image
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Store original image for display
            st.session_state.current_original_image = image
            st.session_state.image_source = 'upload'
            
            # Preprocess for model
            processed_image = preprocess_image_for_model(image_array, 'efficientnetb0')
            
            return processed_image
            
        except Exception as e:
            render_error_message(f"Image preprocessing error: {str(e)}")
            return None
    
    def preprocess_image_from_camera(self, camera_image):
        """Preprocess image from camera"""
        try:
            # Clear previous results
            st.session_state.current_prediction = None
            st.session_state.current_image_array = None
            st.session_state.current_original_image = None
            
            # Convert PIL to numpy array
            image_array = np.array(camera_image)
            
            # Store original image for display
            st.session_state.current_original_image = camera_image
            st.session_state.image_source = 'camera'
            
            # Preprocess for model
            processed_image = preprocess_image_for_model(image_array, 'efficientnetb0')
            
            return processed_image
        except Exception as e:
            st.error(f"Error processing camera image: {str(e)}")
            return None
    
    def render_settings_page(self):
        st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
        
        # Model settings
        st.markdown("### Model Settings")
        
        use_ensemble = st.checkbox(
            "Use Ensemble (Recommended)",
            value=self.use_ensemble
        )
        
        if use_ensemble != self.use_ensemble:
            self.use_ensemble = use_ensemble
            self._load_production_model()
        
        # Display model status
        if self.model_ensemble:
            status = self.model_ensemble.get_model_status()
            st.json(status)
    
    def run(self):
        """Main application runner"""
        # Set page config
        st.set_page_config(
            page_title="Skin Lesion Analysis",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Analysis", "History", "Settings"]
        )
        
        # Render selected page
        if page == "Analysis":
            self.render_analysis_page()
        elif page == "History":
            self.render_history_page()
        elif page == "Settings":
            self.render_settings_page()

# Main execution
if __name__ == "__main__":
    app = ProductionSkinVestigatorApp()
    app.run()
