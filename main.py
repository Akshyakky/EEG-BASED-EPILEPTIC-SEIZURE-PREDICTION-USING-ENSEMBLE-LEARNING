import streamlit as st
from page1 import page_1
from page2 import page_2
from page3 import page_3
from page4 import page_4
from model_definitions import EnhancedEpilepsyModel

def main():
    # Configure the page
    st.set_page_config(
        page_title='EEG-Based Epileptic Seizure Prediction',
        page_icon="⚕️",
        layout="wide"
    )
    
    # Initialize session state for page selection if it doesn't exist
    if 'page_selection' not in st.session_state:
        st.session_state.page_selection = "Home"
    
    # Define a sidebar navigation menu
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.selectbox(
        "Go to", 
        ["Home", "About Epilepsy", "Prediction", "Precautions"],
        index=["Home", "About Epilepsy", "Prediction", "Precautions"].index(st.session_state.page_selection)
    )
    
    # Update session state when sidebar selection changes
    if page_selection != st.session_state.page_selection:
        st.session_state.page_selection = page_selection
        st.rerun()
    
    # Display the selected page based on session state
    if st.session_state.page_selection == "Home":
        page_1()
    elif st.session_state.page_selection == "About Epilepsy":
        page_4()
    elif st.session_state.page_selection == "Prediction":
        page_2()
    elif st.session_state.page_selection == "Precautions":
        page_3()

if __name__ == "__main__":
    main()