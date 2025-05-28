import streamlit as st
from model_module import ModelTrainer
from eda_module import EDAAnalyzer
from load_dataset_module import DataLoader

st.title("Model Training")
st.write("Train machine learning models to predict health outcomes")
st.image("model.jpg", width=300)

# Load your dataset
data_loader = DataLoader("data.csv")
df = data_loader.load_data()

#Set EDA
eda = EDAAnalyzer(df)

# Set ID to Index
df = eda._set_index('ID')

target_options = [
            'Chronic Stress', 
            'Physical Activity', 
            'Income Level', 
            'Stroke Occurrence']

model_options = [
            'Logistic Regression',
            'Decision Tree',
            'Random Forest',
            'Gradient Boosting']


# Target selection
st.subheader("1. Select Prediction Target")
target = st.selectbox(
    "What do you want to predict?",
    target_options,
    key="target_select"
)

# Model selection
st.subheader("2. Select Models")
selected_models = st.multiselect(
    "Choose which algorithms to use:",
    model_options,
    key="model_select"
)

# Training button
# st.subheader("3. Train Models")
if st.button("Start Training") and selected_models and target:
    with st.spinner("Training models..."):
        try:
            trainer = ModelTrainer(df)
            trainer.select_models(selected_models)
            results = trainer.train_for_target(target)
            
            # Display results in tabs
            tabs = st.tabs([f"Model: {name}" for name in results.keys()])
            
            for tab, (_, metrics) in zip(tabs, results.items()):
                with tab:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
                        
                        st.subheader("Confusion Matrix")
                        st.dataframe(metrics["Confusion Matrix"])
                    
                    with col2:
                        st.subheader("Classification Report")
                        st.text(metrics["Classification Report"])
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif not selected_models:
    st.warning("Please select at least one model")
elif not target:
    st.warning("Please select a target variable")

st.divider()
st.write("@CopyRight Umoren, Wisdom Akpabio")