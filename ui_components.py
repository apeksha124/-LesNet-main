"""
UI Components - Streamlit Interface Functions
Extracted from app_production_fixed.py for better organization
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

def render_patient_education(pred):
    """Render patient education section"""
    st.markdown("### Patient Education")
    
    for education in pred['recommendation']['patient_education']:
        st.write(f"**{education}**")

def render_medical_disclaimer():
    """Render medical disclaimer"""
    st.markdown("### Medical Disclaimer")
    st.warning("""
    **Important Medical Notice:**
    - This AI system provides assistance and recommendations, not medical diagnosis
    - Always consult with qualified healthcare providers for diagnosis and treatment
    - Use this tool as a supplement to, not replacement for, professional medical judgment
    - In case of emergency, seek immediate medical attention
    """)

def render_history_page():
    """Render patient history"""
    st.markdown('<h1 class="main-header">Patient Consultation History</h1>', unsafe_allow_html=True)
    
    # Patient selection
    patient_id = st.text_input("Enter Patient ID")
    
    if patient_id:
        consultations = [c for c in st.session_state.get('consultations', []) 
                       if c['patient_id'] == patient_id]
        
        if consultations:
            st.markdown(f"### Consultation History for {patient_id}")
            
            for i, consultation in enumerate(consultations, 1):
                with st.expander(f"Consultation {i} - {consultation['timestamp']} ({consultation.get('image_source', 'unknown').title()})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Risk Level:** {consultation['risk_level']}")
                        st.write(f"**Risk Score:** {consultation.get('risk_score', 0):.2f}")
                        st.write(f"**Confidence:** {consultation['confidence']:.1%}")
                        st.write(f"**Prediction:** {consultation['pred_class']}")
                        st.write(f"**Image Source:** {consultation.get('image_source', 'unknown').title()}")
                        st.write(f"**Grad-CAM:** {'Available' if consultation.get('gradcam_available', False) else 'Not Available'}")
                    
                    with col2:
                        if consultation['symptoms']:
                            st.write("**Symptoms:**")
                            st.write(consultation['symptoms'])

def render_care_recommendations(pred):
    """Render care recommendations"""
    st.markdown("### Care Recommendations")
    
    for care in pred['recommendation']['care']:
        st.write(f"**{care}**")

def render_risk_assessment(pred):
    """Render risk assessment"""
    risk_level = pred['recommendation']['risk_level']
    risk_score = pred['recommendation']['risk_score']
    
    st.markdown("### Risk Assessment")
    
    # Risk level indicator
    if risk_level == 'low':
        st.success(f"**Risk Level: {risk_level.upper()}**")
    elif risk_level == 'medium':
        st.warning(f"**Risk Level: {risk_level.upper()}**")
    else:
        st.error(f"**Risk Level: {risk_level.upper()}**")
    
    st.write(f"**Risk Score:** {risk_score:.2f}")

def render_prediction_summary(pred):
    """Render prediction summary"""
    st.markdown("### Prediction Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Class", pred['class_name'])
    
    with col2:
        st.metric("Confidence", f"{pred['confidence']:.1%}")
    
    with col3:
        st.metric("Risk Level", pred['recommendation']['risk_level'].title())

def render_gradcam_display(gradcam_img, confidence):
    """Render Grad-CAM heatmap"""
    if gradcam_img is not None:
        st.markdown("### Grad-CAM Visualization")
        st.image(gradcam_img, caption=f"Model attention heatmap (Confidence: {confidence:.1%})")
    else:
        st.warning("Grad-CAM visualization not available for this prediction")

def render_image_display(image, caption="Uploaded Image"):
    """Render image display"""
    st.image(image, caption=caption, use_column_width=True)

def render_confidence_bar(confidence):
    """Render confidence as progress bar"""
    st.markdown("### Confidence Level")
    st.progress(confidence)
    st.write(f"Confidence: {confidence:.1%}")

def render_class_probabilities(probabilities, class_names):
    """Render class probabilities as bar chart"""
    st.markdown("### Class Probabilities")
    
    fig, ax = plt.subplots()
    bars = ax.bar(class_names, probabilities)
    ax.set_ylabel('Probability')
    ax.set_title('Model Prediction Probabilities')
    ax.set_ylim(0, 1)
    
    # Color code the bars
    colors = ['green' if p == max(probabilities) else 'lightblue' for p in probabilities]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    st.pyplot(fig)

def render_model_comparison(model_results):
    """Render individual model results comparison"""
    st.markdown("### Individual Model Results")
    
    for i, result in enumerate(model_results):
        with st.expander(f"Model {i+1}: {result['model_name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Prediction:** {result['pred_class']}")
                st.write(f"**Confidence:** {result['confidence']:.1%}")
            
            with col2:
                # Create confidence bar
                st.progress(result['confidence'])
                st.write(f"Model Confidence: {result['confidence']:.1%}")

def render_error_message(error_text):
    """Render error message"""
    st.error(f"Error: {error_text}")
    st.markdown("Please try again or contact support if the issue persists.")

def render_loading_indicator():
    """Render loading indicator"""
    st.info("Processing image... Please wait.")

def render_success_message(message):
    """Render success message"""
    st.success(message)

def render_info_message(message):
    """Render info message"""
    st.info(message)

def render_warning_message(message):
    """Render warning message"""
    st.warning(message)

def format_confidence_for_display(confidence):
    """Format confidence for display"""
    return f"{confidence:.1%}"

def format_risk_score_for_display(risk_score):
    """Format risk score for display"""
    return f"{risk_score:.2f}"

def get_risk_color(risk_level):
    """Get color for risk level"""
    if risk_level == 'low':
        return 'green'
    elif risk_level == 'medium':
        return 'orange'
    else:
        return 'red'

def render_ensemble_info(ensemble_details):
    """Render ensemble information"""
    st.markdown("### Ensemble Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Agreement Rate:** {ensemble_details['agreement_rate']:.1%}")
        st.write(f"**Models Used:** {len(ensemble_details['model_results'])}")
    
    with col2:
        st.write(f"**Ensemble Class:** {ensemble_details['class']}")
        st.write(f"**Confidence Variance:** {ensemble_details.get('confidence_variance', 0):.4f}")

def render_consultation_summary(consultation):
    """Render single consultation summary"""
    st.markdown("### Consultation Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Patient ID:** {consultation.get('patient_id', 'N/A')}")
        st.write(f"**Timestamp:** {consultation.get('timestamp', 'N/A')}")
        st.write(f"**Image Source:** {consultation.get('image_source', 'N/A').title()}")
    
    with col2:
        st.write(f"**Prediction:** {consultation.get('pred_class', 'N/A')}")
        st.write(f"**Confidence:** {consultation.get('confidence', 0):.1%}")
        st.write(f"**Risk Level:** {consultation.get('risk_level', 'N/A')}")

def render_css_styles():
    """Render custom CSS styles"""
    st.markdown("""
    <style>
    .main-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .result-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
    }
    .high-risk {
        border-left: 4px solid #ff4b4b;
    }
    .medium-risk {
        border-left: 4px solid #ffa500;
    }
    .low-risk {
        border-left: 4px solid #00c851;
    }
    </style>
    """, unsafe_allow_html=True)
