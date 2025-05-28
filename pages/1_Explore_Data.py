import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from eda_module import EDAAnalyzer
from load_dataset_module import DataLoader

# Page Interface
st.title("Data Exploration")
st.write("Explore the dataset and visualize distributions")
st.image("analysis.jpg", width=300)

# Load your dataset
data_loader = DataLoader("data.csv")
df = data_loader.load_data()

#Set EDA
eda = EDAAnalyzer(df)

# Set ID to Index
df = eda._set_index('ID')

#update EDA
eda = EDAAnalyzer(df)

with st.expander("Basic Information", expanded=False):
    if st.button("Show Dataset Info", key="dataset_info"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Dataset Shape:")
            st.write(eda.df.shape)
            
            st.text("\nData Types:")
            st.write(eda.df.dtypes)
        
        with col2:
            st.write("#")
            st.text("\nMissing Values:")
            st.write(eda.df.isnull().sum())

with st.expander("Target Distributions", expanded=False):
    if st.button("Show Target Distributions", key="target_dist"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Chronic Stress (Binary)
        eda.df['Chronic Stress'].value_counts().plot(
            kind='bar', ax=axes[0,0], title='Chronic Stress (0/1)')
        
        # Physical Activity (Ordinal)
        activity_order = sorted(eda.df['Physical Activity'].unique())
        eda.df['Physical Activity'].value_counts().loc[activity_order].plot(
            kind='bar', ax=axes[0,1], title='Physical Activity')
        
        # Income Level (Ordinal)
        income_order = sorted(eda.df['Income Level'].unique())
        eda.df['Income Level'].value_counts().loc[income_order].plot(
            kind='bar', ax=axes[1,0], title='Income Level')
        
        # Stroke Occurrence (Binary)
        eda.df['Stroke Occurrence'].value_counts().plot(
            kind='bar', ax=axes[1,1], title='Stroke Occurrence (0/1)')
        
        plt.tight_layout()
        st.pyplot(fig)

with st.expander("Feature Distributions", expanded=False):
    features = [col for col in eda.df.columns if col not in eda.target_columns]
    selected_feature = st.selectbox("Select a feature to visualize", features)
    
    fig, ax = plt.subplots(figsize=(8,4))
    if eda.df[selected_feature].nunique() < 10:
        eda.df[selected_feature].value_counts().plot(kind='bar', ax=ax)
    else:
        eda.df[selected_feature].hist(bins=30, ax=ax)
    ax.set_title(f'Distribution of {selected_feature}')
    st.pyplot(fig)

with st.expander("Correlation Analysis", expanded=False):
    if st.button("Show Correlations", key="correlations"):
        st.text("=== Correlation Analysis (Numerical Features Only) ===")
        
        # Select only numerical features (excluding targets)
        numerical_cols = eda.df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col not in eda.target_columns]
        
        if not numerical_cols:
            st.warning("No numerical features found for correlation analysis")
        else:
            # Create a numerical-only dataframe
            df_numerical = eda.df[numerical_cols].copy()
            
            # Add encoded versions of ordinal targets if they're not already numerical
            if 'Physical Activity' in eda.df.columns:
                df_numerical['Physical Activity_encoded'] = pd.factorize(eda.df['Physical Activity'])[0]
            if 'Income Level' in eda.df.columns:
                df_numerical['Income Level_encoded'] = pd.factorize(eda.df['Income Level'])[0]
            
            # Add binary targets directly
            for target in ['Chronic Stress', 'Stroke Occurrence']:
                if target in eda.df.columns:
                    df_numerical[target] = eda.df[target]
            
            # Calculate and display correlations
            corr_matrix = df_numerical.corr()
            
            # Display correlations for each target
            for target in eda.target_columns:
                if target in df_numerical.columns or f"{target}_encoded" in df_numerical.columns:
                    st.subheader(f"Correlations with {target}:")
                    target_col = target if target in df_numerical.columns else f"{target}_encoded"
                    st.dataframe(
                        corr_matrix[target_col]
                        .sort_values(ascending=False)
                        .drop(index=eda.target_columns, errors='ignore')
                        .head(10)
                    )
st.divider()
st.write("@CopyRight Umoren, Wisdom Akpabio")