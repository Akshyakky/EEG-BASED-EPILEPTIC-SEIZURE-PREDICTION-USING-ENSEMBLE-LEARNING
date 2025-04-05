import streamlit as st

def page_3():
    """Precautions page with guidance on managing seizure risks"""
    
    st.title("Seizure Precautions and Management")
    
    st.markdown("""
    ## Immediate Steps When a Seizure is Predicted
    
    When the system predicts a high likelihood of a seizure, taking prompt action can help reduce risks 
    and ensure safety. Below are recommended steps to follow:
    """)
    
    # Create expandable sections for different types of precautions
    with st.expander("🔍 Recognize Early Warning Signs", expanded=True):
        st.markdown("""
        Many people experience "auras" or warning signs before a seizure occurs:
        
        - Unusual smells, tastes, or sensations
        - Visual changes or hallucinations
        - Sudden mood changes
        - Dizziness or lightheadedness
        - Nausea
        - Tingling or numbness
        
        If you notice these symptoms along with a prediction alert, take immediate precautions.
        """)
    
    with st.expander("🛌 Find a Safe Place", expanded=True):
        st.markdown("""
        - Move away from dangerous areas (stairs, roads, sharp objects)
        - Sit or lie down in a safe, cushioned area if possible
        - Clear the area of hazardous objects
        - Loosen tight clothing, especially around the neck
        """)
    
    with st.expander("💊 Medication Protocol", expanded=True):
        st.markdown("""
        - If prescribed rescue medication (such as a benzodiazepine), consider taking it according to your doctor's instructions
        - Do not take extra doses of daily medication unless specifically instructed by your healthcare provider
        - Keep track of any medication taken to report to your healthcare team
        """)
    
    with st.expander("👥 Alert Support System", expanded=True):
        st.markdown("""
        - Notify a family member, caregiver, or friend about the prediction
        - If alone, consider calling someone to check on you
        - Activate any medical alert systems if available
        - Consider using seizure monitoring apps that can alert others
        """)
    
    st.markdown("""
    ## During a Seizure (For Caregivers)
    """)
    
    with st.expander("✋ First Aid for Seizures", expanded=True):
        st.markdown("""
        If you are with someone having a seizure:
        
        - **DO** cushion their head
        - **DO** loosen tight clothing around the neck
        - **DO** turn them onto their side after movements subside (recovery position)
        - **DO** time the seizure
        - **DO NOT** restrain the person
        - **DO NOT** put anything in their mouth
        - **DO NOT** give food, drink, or medication until fully alert
        """)
    
    with st.expander("🚑 When to Call Emergency Services", expanded=True):
        st.markdown("""
        Call emergency services (911) if:
        
        - The seizure lasts longer than 5 minutes
        - The person doesn't wake up after the seizure stops
        - The person has another seizure shortly after the first
        - The person is injured during the seizure
        - The person has breathing difficulties
        - The person has never had a seizure before
        - The person is pregnant or has diabetes
        """)
    
    st.markdown("""
    ## Long-term Management
    """)
    
    with st.expander("📝 Seizure Diary", expanded=True):
        st.markdown("""
        Maintain a seizure diary recording:
        
        - Date and time of prediction alerts
        - Whether a seizure occurred following the alert
        - Duration and characteristics of any seizures
        - Potential triggers noticed
        - Effectiveness of any interventions tried
        
        This information is valuable for your healthcare provider to adjust treatment plans.
        """)
    
    with st.expander("🧠 Lifestyle Adjustments", expanded=True):
        st.markdown("""
        Common seizure triggers to manage:
        
        - **Sleep**: Maintain regular sleep patterns and get adequate rest
        - **Stress**: Practice stress-reduction techniques like meditation
        - **Diet**: Avoid skipping meals and stay hydrated
        - **Alcohol**: Limit or avoid alcohol consumption
        - **Screen time**: Reduce exposure to flashing lights if photosensitive
        - **Medications**: Take as prescribed and avoid missing doses
        """)
    
    st.markdown("""
    ---
    
    ### Disclaimer
    
    This information is provided for educational purposes only and is not a substitute for 
    professional medical advice. Always consult with your neurologist or epilepsy specialist 
    for personalized guidance on managing your condition.
    """)

if __name__ == "__main__":
    # For testing this page directly
    page_3()