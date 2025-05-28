import pandas as pd
import matplotlib.pyplot as plt

class EDAAnalyzer:
    def __init__(self, df):
        self.df = df
        self.target_columns = [
            'Chronic Stress', 
            'Physical Activity', 
            'Income Level', 
            'Stroke Occurrence'
        ]

    def _set_index(self, col):
        self.df = self.df.set_index(col)
        return self.df
        
    def basic_info(self):
        """Show basic dataset information"""
        print(f"Dataset Shape: {self.df.shape}")
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
    def plot_target_distributions(self):
        """Visualize all target distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Chronic Stress (Binary)
        self.df['Chronic Stress'].value_counts().plot(
            kind='bar', ax=axes[0,0], title='Chronic Stress (0/1)')
        
        # Physical Activity (Ordinal)
        activity_order = sorted(self.df['Physical Activity'].unique())
        self.df['Physical Activity'].value_counts().loc[activity_order].plot(
            kind='bar', ax=axes[0,1], title='Physical Activity')
        
        # Income Level (Ordinal)
        income_order = sorted(self.df['Income Level'].unique())
        self.df['Income Level'].value_counts().loc[income_order].plot(
            kind='bar', ax=axes[1,0], title='Income Level')
        
        # Stroke Occurrence (Binary)
        self.df['Stroke Occurrence'].value_counts().plot(
            kind='bar', ax=axes[1,1], title='Stroke Occurrence (0/1)')
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_distributions(self):
        """Plot distributions of non-target features"""
        features = [col for col in self.df.columns if col not in self.target_columns]
        
        for feature in features:
            plt.figure(figsize=(8,4))
            if self.df[feature].nunique() < 10:
                self.df[feature].value_counts().plot(kind='bar')
            else:
                self.df[feature].hist(bins=30)
            plt.title(f'Distribution of {feature}')
            plt.show()
            
    def analyze_correlations(self):
        """Safe correlation analysis that only uses numerical features"""
        print("\n=== Correlation Analysis (Numerical Features Only) ===")
        
        # Select only numerical features (excluding targets)
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col not in self.target_columns]
        
        if not numerical_cols:
            print("No numerical features found for correlation analysis")
            return
            
        # Create a numerical-only dataframe
        df_numerical = self.df[numerical_cols].copy()
        
        # Add encoded versions of ordinal targets if they're not already numerical
        if 'Physical Activity' in self.df.columns:
            df_numerical['Physical Activity_encoded'] = pd.factorize(self.df['Physical Activity'])[0]
        if 'Income Level' in self.df.columns:
            df_numerical['Income Level_encoded'] = pd.factorize(self.df['Income Level'])[0]
        
        # Add binary targets directly
        for target in ['Chronic Stress', 'Stroke Occurrence']:
            if target in self.df.columns:
                df_numerical[target] = self.df[target]
        
        # Calculate and display correlations
        corr_matrix = df_numerical.corr()
        
        # Print correlations for each target
        for target in self.target_columns:
            if target in df_numerical.columns or f"{target}_encoded" in df_numerical.columns:
                print(f"\nCorrelations with {target}:")
                target_col = target if target in df_numerical.columns else f"{target}_encoded"
                print(corr_matrix[target_col].sort_values(ascending=False).drop(index=self.target_columns, errors='ignore').head(10))