import streamlit as st

def page_1():
    """Home page for the epilepsy prediction application"""
    
    st.title("EEG-Based Epileptic Seizure Prediction")
    
    # Add a hero image if available
    # st.image("hero_image.jpg", use_column_width=True)
    
    st.markdown("""
    ## Welcome to the Epileptic Seizure Prediction System
    
    This application uses advanced machine learning techniques to analyze Electroencephalogram (EEG) 
    data and predict the likelihood of an epileptic seizure. Early prediction can help patients and 
    caregivers take preventive measures, potentially reducing the impact of seizures and improving 
    quality of life.
    
    ### Key Features:
    
    - **Real-time EEG Data Analysis**: Input EEG channel readings for immediate seizure prediction
    - **Precautionary Guidance**: Receive guidance on steps to take when a seizure is predicted
    - **Educational Resources**: Learn about epilepsy, its causes, symptoms, and management
    
    ### How to Use This Application:
    
    1. Navigate to the **Prediction** page to input EEG readings and get predictions
    2. Visit the **About Epilepsy** page to learn more about the condition
    3. Check the **Precautions** page for guidance on managing seizure risks
    
    ### Get Started:
    """)
    
    # Create buttons for navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Make a Prediction"):
            st.session_state.page_selection = "Prediction"
            st.rerun()
    
    with col2:
        if st.button("Learn About Epilepsy"):
            st.session_state.page_selection = "About Epilepsy"
            st.rerun()
    
    with col3:
        if st.button("View Precautions"):
            st.session_state.page_selection = "Precautions"
            st.rerun()
    
    st.markdown("""
    ---
    
    ### Disclaimer
    
    This application is designed as a supportive tool and should not replace professional medical 
    advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for 
    medical concerns.
    """)

if __name__ == "__main__":
    # For testing this page directly
    page_1()