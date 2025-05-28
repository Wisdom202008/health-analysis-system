import streamlit as st

st.set_page_config("Disease Prediction System", layout="wide", page_icon=":house:") 

st.title("Cardiovascular Disease Prediction System: Implementation and Analysis")

col1, col2 = st.columns(2)
        
with col1:
        st.markdown("""    
        Cardiovascular disease (CVD) remains one of the leading causes of death globally, and the UK National Health Service (NHS) identifies it as a top public health priority.
        The convergence of digital health technology with traditional healthcare practice offers heartening potential for preventing cardiovascular disease fatalities.
                    
        With the application of patient data like vital signs and lifestyle factors, healthcare practitioners can assist in the identification of susceptible individuals and implement preventive care measures prior to
        life-threatening events occurring. This project addresses this challenge by developing an end-to-end cardiovascular disease prediction tool that has been trained on a 172,000-simulated patient
        record dataset.

        """)

with col2:
        st.image("smartphone-earth-with-stethoscope-heart.jpg",width=500)

st.divider()

col1, col2 = st.columns([1,3.5])

with col1:
    st.image("african-american-doctor.jpg",width=200)

with col2:

    st.markdown("""           
    ### About This Project      
    The tool forecasts four significant outcomes:
    - Chronic Stress
    - Physical Activity
    - Income Level
    - Stroke Occurrence
                
    The implementation employs object-oriented programming practices to generate a modular, maintainable solution with data preprocessing, exploratory data analysis, machine learning model training, and a user interface for interaction.

    """)

st.divider()


st.markdown("""   
    ### Features:
    - **Data Exploration**: Visualize distributions and relationships in the data
    - **Model Training**: Train machine learning models to predict health outcomes   
    """)
      
st.divider()
st.write("@CopyRight Umoren, Wisdom Akpabio")