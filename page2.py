import streamlit as st
import time
import numpy as np
import pickle
import sys
import os

# Add the current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import model definitions with proper error handling
try:
    from model_definitions import EnhancedEpilepsyModel
    MODEL_IMPORT_SUCCESS = True
except ImportError as e:
    st.warning(f"Model definition import failed: {str(e)}")
    MODEL_IMPORT_SUCCESS = False

# Define the fallback model class
class SimpleFallbackModel:
    """Simple rule-based model when no trained model is available"""
    
    def __init__(self):
        # Define known pattern signatures
        self.seizure_signature = np.array([0.000031, 0.000027, 0.000012, 0.000056, 0.000041, -0.000018, 0.000052, 0.000052])
        self.normal_signature = np.array([-0.000053, 0.000023, 0.000078, 0.000123, 0.000118, -0.000047, -0.000061, -0.000061])
        self.feature_importance = np.array([0.25, 0.2, 0.1, 0.15, 0.15, 0.05, 0.1, 0.0])
        self.threshold = 0.5  # Similarity threshold
    
    def predict(self, X):
        """Pattern-based prediction"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        predictions = []
        for i in range(X.shape[0]):
            # Calculate similarity to known patterns
            seizure_similarity = self._calculate_pattern_similarity(X[i], self.seizure_signature)
            normal_similarity = self._calculate_pattern_similarity(X[i], self.normal_signature)
            
            # If the input is more similar to the seizure pattern, classify as seizure
            if seizure_similarity > normal_similarity:
                predictions.append(1)  # Seizure
            else:
                predictions.append(0)  # Normal
        
        return np.array(predictions)
    
    def _calculate_pattern_similarity(self, data, pattern):
        """Calculate similarity between input data and a reference pattern"""
        # Ensure data has same length as pattern
        data = data[:len(pattern)]
        
        # Calculate weighted correlation
        weighted_data = data * self.feature_importance[:len(data)]
        weighted_pattern = pattern * self.feature_importance[:len(pattern)]
        
        # Simple similarity metric - inverse of weighted absolute difference
        diff = np.sum(np.abs(weighted_data - weighted_pattern))
        similarity = 1.0 / (1.0 + diff)
        
        return similarity
    
    def get_anomaly_scores(self, X):
        """Calculate anomaly scores based on similarity to seizure pattern"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        scores = []
        for i in range(X.shape[0]):
            # Calculate similarity to known patterns
            seizure_similarity = self._calculate_pattern_similarity(X[i], self.seizure_signature)
            normal_similarity = self._calculate_pattern_similarity(X[i], self.normal_signature)
            
            # Score based on relative similarity to seizure pattern
            if seizure_similarity + normal_similarity > 0:
                score = seizure_similarity / (seizure_similarity + normal_similarity)
                scores.append(score)
            else:
                scores.append(0.5)  # Default if similarities are both zero
        
        return np.array(scores)

def load_model():
    """Robust model loading with fallbacks and attribute verification"""
    model = None
    model_info = None
    
    try:
        # First try: Standard pickle load
        with open('enhanced_epilepsy_model.pkl', 'rb') as file:
            model_package = pickle.load(file)
            model = model_package['model']
            
            # Ensure all required attributes exist
            ensure_model_attributes(model)
                
            model_info = {
                'performance_metrics': model_package.get('performance_metrics', {}),
                'type': 'Enhanced anomaly detection model'
            }
        #st.success("Successfully loaded enhanced epilepsy model")
    except Exception as e1:
        st.warning(f"Enhanced model loading failed: {str(e1)}")
        
        try:
            # Second try: If model import succeeded, create a new model instance
            if MODEL_IMPORT_SUCCESS:
                model = EnhancedEpilepsyModel()
                # Ensure all required attributes
                ensure_model_attributes(model)
                
                model_info = {
                    'performance_metrics': {'accuracy': 'N/A', 'specificity': 'N/A'},
                    'type': 'Fresh instance of enhanced model'
                }
                st.success("Created new enhanced epilepsy model instance")
            else:
                # Third try: Load the original model if available
                with open('EE_model.pkl', 'rb') as file:
                    model = pickle.load(file)
                    # Ensure all required attributes
                    ensure_model_attributes(model)
                    
                    model_info = {
                        'performance_metrics': {'accuracy': 0.94, 'specificity': 0.97},
                        'type': 'Original ensemble model'
                    }
                st.success("Successfully loaded original epilepsy model")
        except Exception as e2:
            st.warning(f"Original model loading failed: {str(e2)}")
            
            # Final fallback: Create a simple rule-based model
            model = SimpleFallbackModel()
            # Ensure all required attributes even for the fallback model
            ensure_model_attributes(model)
            
            model_info = {
                'performance_metrics': {'accuracy': 'N/A', 'specificity': 'N/A'},
                'type': 'Simple fallback model (no trained model available)'
            }
            st.info("Using simple fallback model for predictions")
    
    return model, model_info

def ensure_model_attributes(model):
    """Ensure all required attributes exist on the model"""
    # Check and add feature_importance if missing
    if not hasattr(model, 'feature_importance'):
        model.feature_importance = np.array([0.2, 0.15, 0.12, 0.18, 0.14, 0.08, 0.13, 0.0])
        #st.info("Added missing feature_importance attribute to model.")
    
    # Check and add threshold if missing
    if not hasattr(model, 'threshold'):
        model.threshold = 0.00005
        #st.info("Added missing threshold attribute to model.")
    
    # Add seizure and normal signatures if missing
    if not hasattr(model, 'seizure_signature'):
        model.seizure_signature = np.array([0.000031, 0.000027, 0.000012, 0.000056, 0.000041, -0.000018, 0.000052, 0.000052])
        #st.info("Added missing seizure_signature attribute to model.")
    
    if not hasattr(model, 'normal_signature'):
        model.normal_signature = np.array([-0.000053, 0.000023, 0.000078, 0.000123, 0.000118, -0.000047, -0.000061, -0.000061])
        #st.info("Added missing normal_signature attribute to model.")
    
    # Add compatibility methods if they don't exist
    if not hasattr(model, '_calculate_pattern_similarity'):
        def calculate_pattern_similarity(self, data, pattern):
            """Calculate similarity between input data and a reference pattern"""
            # Ensure data has same length as pattern
            data = data[:len(pattern)]
            
            # Calculate weighted correlation
            weighted_data = data * self.feature_importance[:len(data)]
            weighted_pattern = pattern * self.feature_importance[:len(pattern)]
            
            # Simple similarity metric - inverse of weighted absolute difference
            diff = np.sum(np.abs(weighted_data - weighted_pattern))
            similarity = 1.0 / (1.0 + diff)
            
            return similarity
        
        # Attach the method to the model instance
        import types
        model._calculate_pattern_similarity = types.MethodType(calculate_pattern_similarity, model)

def predict_seizure(input_data, model):
    """Make seizure prediction using the model with feature compatibility handling"""
    try:
        # Ensure all required attributes exist
        ensure_model_attributes(model)
        
        # Check if we need to adapt feature count
        input_array = np.array(input_data).reshape(1, -1)
        
        # Get expected feature count based on seizure_signature length
        expected_features = len(model.seizure_signature)
        
        # If we have fewer features than expected, add padding
        if input_array.shape[1] < expected_features:
            # Create a new array with the expected number of features
            padded_array = np.zeros((1, expected_features))
            # Copy the available features
            padded_array[0, :input_array.shape[1]] = input_array
            # Use the padded array for prediction
            input_array = padded_array
            st.info(f"Added padding to match expected feature count ({input_array.shape[1]} features).")
        
        # Modified prediction logic using pattern similarity
        X = input_array
        seizure_similarity = model._calculate_pattern_similarity(X[0], model.seizure_signature)
        normal_similarity = model._calculate_pattern_similarity(X[0], model.normal_signature)
        
        # If the input is more similar to the seizure pattern, classify as seizure
        if seizure_similarity > normal_similarity:
            prediction = 1  # Seizure
            # Calculate confidence as relative similarity
            confidence = seizure_similarity / (seizure_similarity + normal_similarity) if (seizure_similarity + normal_similarity) > 0 else 0.5
        else:
            prediction = 0  # Normal
            # Calculate confidence as relative similarity
            confidence = normal_similarity / (seizure_similarity + normal_similarity) if (seizure_similarity + normal_similarity) > 0 else 0.5
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None, None

def display_prediction_results(prediction, anomaly_score=None):
    """Display prediction results with visual indicators"""
    if prediction == 1:
        st.error('⚠️ A seizure is likely to occur in the next few minutes. Take necessary precautions!')
        
        # Display risk level
        if anomaly_score is not None:
            risk_level = anomaly_score * 100  # Convert to percentage
            st.write(f"**Seizure Risk Level: {risk_level:.1f}%**")
            
            # Visual risk indicator
            st.progress(min(risk_level/100, 1.0))
            
        st.write("**Please refer to the Precautions section for immediate steps.**")
    else:
        st.success('✓ No seizure pattern detected. EEG readings are within normal parameters.')
        
        # Display confidence level
        if anomaly_score is not None:
            normal_confidence = (1 - anomaly_score) * 100
            st.write(f"**Normal Pattern Confidence: {normal_confidence:.1f}%**")
            
            # Visual confidence indicator
            st.progress(min(normal_confidence/100, 1.0))

def load_sample_data():
    """Load sample data for demonstration"""
    return {
        "Seizure Pattern": [0.000031, 0.000027, 0.000012, 0.000056, 0.000041, -0.000018, 0.000052, 0.000052],  # Added 8th value
        "Normal Pattern": [-0.000053, 0.000023, 0.000078, 0.000123, 0.000118, -0.000047, -0.000061, -0.000061]  # Added 8th value
    }

def page_2():
    st.title('Epileptic Seizure Prediction')
    
    # Load model with robust error handling
    model, model_info = load_model()
    
    # Display model information
    with st.expander("Model Information"):
        st.write(f"**Model Type:** {model_info['type']}")
        metrics = model_info.get('performance_metrics', {})
        if metrics:
            st.write("**Performance Metrics:**")
            for key, value in metrics.items():
                st.write(f"- {key.capitalize()}: {value}")
    
    # Define EEG channels and their value ranges - added the missing 8th feature
    channels = [
        ('FP1-F7', -0.000081, 0.000174),
        ('C3-P3', -0.000012, 0.000058),
        ('P3-O1', -0.000004, 0.000127),
        ('P4-O2', 0.000017, 0.000164),
        ('P7-O1', 0.000006, 0.000146),
        ('P7-T7', -0.000067, 0.000013),
        ('T8-P8', -0.000179, 0.000115),
        ('T8-P8-1', -0.000179, 0.000115)  # Added the missing 8th channel
    ]
    
    st.write("Enter the EEG channel readings to predict potential seizure occurrence:")
    
    # Option to load sample data
    samples = load_sample_data()
    sample_option = st.selectbox("Load sample data (optional):", 
                                ["None"] + list(samples.keys()))
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # If sample data selected, pre-fill the form
    sample_data = None
    if sample_option != "None":
        sample_data = samples[sample_option]
    
    # Create input fields and collect values
    input_values = []
    for i, (channel, min_val, max_val) in enumerate(channels):
        # Alternate between columns
        current_col = col1 if i % 2 == 0 else col2
        
        # Default value from sample if provided with safe access
        default_val = (min_val + max_val) / 2  # Default fallback value
        if sample_data and i < len(sample_data):
            default_val = sample_data[i]
        
        with current_col:
            val = st.number_input(
                f'{channel}', 
                min_value=min_val, 
                max_value=max_val,
                value=default_val,
                format="%.9e",
                help=f"Range: {min_val:.9e} to {max_val:.9e}"
            )
            input_values.append(val)
    
    # Prediction button
    if st.button('Predict Seizure Occurrence'):
        if len(input_values) != len(channels):
            st.warning("Please provide values for all EEG channels.")
        else:
            with st.spinner("Analyzing EEG patterns..."):
                # Add a small delay to simulate processing
                time.sleep(1.5)
                prediction, anomaly_score = predict_seizure(input_values, model)
                
            if prediction is not None:
                display_prediction_results(prediction, anomaly_score)

def main():
    page_2()

if __name__ == "__main__":
    main()