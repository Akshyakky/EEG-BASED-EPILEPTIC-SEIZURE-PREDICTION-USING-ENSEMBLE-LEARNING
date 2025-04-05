import streamlit as st

def page_4():
    """Educational page about epilepsy"""
    
    st.title("Understanding Epilepsy")
    
    st.markdown("""
    ## What is Epilepsy?
    
    Epilepsy is a neurological disorder characterized by recurrent seizures. A seizure is a sudden, 
    uncontrolled electrical disturbance in the brain that can cause changes in behavior, movements, 
    feelings, and levels of consciousness.
    
    Approximately 50 million people worldwide have epilepsy, making it one of the most common 
    neurological diseases globally.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Types of Seizures", 
        "Causes", 
        "Diagnosis", 
        "Treatment", 
        "Living with Epilepsy"
    ])
    
    with tab1:
        st.markdown("""
        ### Types of Seizures
        
        Seizures are classified into two main categories:
        
        #### Focal (Partial) Seizures
        These seizures begin in one area of the brain:
        
        - **Simple Focal Seizures**: Affect a small part of the brain and may cause twitching or changes in sensation like unusual tastes or smells. Consciousness is not affected.
        
        - **Complex Focal Seizures**: Affect a larger area of the brain and can cause confusion or impaired awareness. The person may stare blankly or perform repetitive movements.
        
        #### Generalized Seizures
        These seizures affect both sides of the brain at once:
        
        - **Absence Seizures**: Brief episodes of staring and impaired awareness.
        
        - **Tonic-Clonic Seizures**: Formerly known as "grand mal" seizures, these cause stiffening of the body (tonic phase) followed by jerking movements (clonic phase) and loss of consciousness.
        
        - **Atonic Seizures**: Cause a sudden loss of muscle tone, often resulting in falls.
        
        - **Myoclonic Seizures**: Brief, shock-like jerks of muscles.
        """)
    
    with tab2:
        st.markdown("""
        ### Causes of Epilepsy
        
        In about half of all cases, the cause of epilepsy is unknown. For others, these factors may play a role:
        
        - **Genetic Factors**: Some types of epilepsy run in families.
        
        - **Brain Injuries**: Trauma from accidents, strokes, or severe infections.
        
        - **Developmental Disorders**: Conditions present at birth that affect the brain.
        
        - **Brain Abnormalities**: Tumors or vascular malformations.
        
        - **Infectious Diseases**: Meningitis, AIDS, viral encephalitis.
        
        - **Prenatal Injury**: Oxygen deficiency to the baby's brain during development.
        
        - **Metabolic Disorders**: Abnormal levels of substances like blood sugar that can affect brain function.
        """)
    
    with tab3:
        st.markdown("""
        ### Diagnosing Epilepsy
        
        Diagnosis typically involves:
        
        #### Medical History and Physical Examination
        Your doctor will ask about your symptoms and medical history, and may perform a neurological exam.
        
        #### Electroencephalogram (EEG)
        This is the most common test for epilepsy, recording the brain's electrical activity. Abnormal patterns can indicate epilepsy.
        
        #### Brain Imaging
        - **MRI (Magnetic Resonance Imaging)**: Provides detailed images of the brain.
        - **CT Scan (Computed Tomography)**: Uses X-rays to create images of the brain.
        
        #### Blood Tests
        To check for infections, genetic disorders, or other conditions that may be causing seizures.
        
        #### Neuropsychological Tests
        Assess thinking, memory, and speech skills.
        """)
    
    with tab4:
        st.markdown("""
        ### Treatment Options
        
        Epilepsy treatment aims to control seizures and depends on the type and severity of seizures.
        
        #### Anti-Seizure Medications
        The most common treatment, these medications can control seizures in about 70% of people with epilepsy.
        
        #### Surgery
        If medications don't work and seizures originate in a well-defined area of the brain, surgery to remove that area may be an option.
        
        #### Vagus Nerve Stimulation (VNS)
        A device implanted under the skin sends electrical impulses to the vagus nerve, which can reduce seizure frequency.
        
        #### Responsive Neurostimulation
        Devices that detect abnormal electrical activity and deliver electrical stimulation to prevent seizures.
        
        #### Ketogenic Diet
        A diet high in fats and low in carbohydrates that can help control seizures in some people, especially children.
        
        #### Deep Brain Stimulation (DBS)
        Electrodes implanted in specific brain areas to regulate electrical activity.
        """)
    
    with tab5:
        st.markdown("""
        ### Living with Epilepsy
        
        Managing epilepsy goes beyond medical treatment:
        
        #### Seizure Triggers
        Identifying and avoiding personal triggers such as:
        - Stress
        - Sleep deprivation
        - Alcohol or drug use
        - Flickering lights (for those with photosensitive epilepsy)
        - Skipping meals
        - Missed medication
        
        #### Safety Precautions
        - Wearing medical alert jewelry
        - Sharing information with family, friends, and colleagues
        - Taking precautions for activities like swimming, bathing, and cooking
        - Considering adaptive equipment for the home
        
        #### Support Systems
        - Joining epilepsy support groups
        - Connecting with epilepsy advocacy organizations
        - Seeking counseling if needed for emotional impact
        
        #### Technology Solutions
        - Seizure detection devices
        - Smartphone apps for tracking seizures and medication
        - Predictive algorithms (like this application) that may help anticipate seizures
        """)
    
    st.markdown("""
    ## The Role of EEG in Epilepsy
    
    ### What is an EEG?
    
    An Electroencephalogram (EEG) records electrical activity in the brain using small metal discs 
    (electrodes) attached to the scalp. Neurons in the brain communicate via electrical impulses, and 
    this activity can be detected and recorded by an EEG machine.
    
    ### How EEG Helps in Epilepsy
    
    - **Diagnosis**: EEG can show abnormal brain wave patterns characteristic of epilepsy
    
    - **Classification**: Helps determine the type of seizures and epilepsy syndrome
    
    - **Localization**: Can help identify the area of the brain where seizures originate
    
    - **Monitoring**: Used for long-term monitoring of brain activity
    
    - **Prediction**: Advanced analysis of EEG data can potentially predict seizure likelihood
    
    The prediction system in this application uses machine learning algorithms to analyze EEG patterns 
    and identify signatures that may precede seizure activity, potentially providing an early warning 
    system for people with epilepsy.
    """)
    
    st.markdown("""
    ---
    
    ### Resources
    
    For more information about epilepsy, visit:
    
    - [Epilepsy Foundation](https://www.epilepsy.com/)
    - [World Health Organization - Epilepsy](https://www.who.int/news-room/fact-sheets/detail/epilepsy)
    - [Centers for Disease Control and Prevention - Epilepsy](https://www.cdc.gov/epilepsy/)
    """)

if __name__ == "__main__":
    # For testing this page directly
    page_4()