import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.base import clone

class ModelTrainer:
    def __init__(self, df):
        """
        Initialize with dataframe and predefined target mappings
        """
        self.df = df
        self.preprocessor = None
        self.random_state = 42
        self.test_size = 0.3
        
        # Dynamic ordinal encoding
        self.target_encoders = {
            'Chronic Stress': None,
            'Physical Activity': OrdinalEncoder(
                categories=[sorted(df['Physical Activity'].unique())],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ),
            'Income Level': OrdinalEncoder(
                categories=[sorted(df['Income Level'].unique())],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ),
            'Stroke Occurrence': None
        }
        
        # Model configurations
        self.available_models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=5,
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                random_state=self.random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
        }
        self.selected_models = []

    def select_models(self, model_names):
        """Select which models to use for training"""
        invalid_models = [name for name in model_names 
                         if name not in self.available_models]
        if invalid_models:
            raise ValueError(f"Invalid models: {invalid_models}. Available: {list(self.available_models.keys())}")
        self.selected_models = model_names

    def prepare_target(self, target_name, y):
        """Encode target variable if needed"""
        if target_name in ['Physical Activity', 'Income Level']:
            encoder = self.target_encoders[target_name]
            return encoder.fit_transform(y.values.reshape(-1, 1)).ravel()
        return y

    def prepare_data(self, target_name):
        """Prepare data with proper preprocessing"""
        if target_name not in self.df.columns:
            raise ValueError(f"Target '{target_name}' not in dataframe")
            
        X = self.df.drop(columns=[
            'Chronic Stress', 'Physical Activity',
            'Income Level', 'Stroke Occurrence'
        ])
        y = self.df[target_name]
        
        # Encode target
        y_encoded = self.prepare_target(target_name, y)
        
        # Identify feature types
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        
        # Split data
        return train_test_split(
            X, y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_state
        )

    def _handle_imbalance(self, X, y, target_name):
        """Handle class imbalance after preprocessing"""
        if target_name in ['Chronic Stress', 'Stroke Occurrence']:
            class_counts = pd.Series(y).value_counts()
            if len(class_counts) == 2 and class_counts.min()/class_counts.max() < 0.5:
                print(f"Applying SMOTE to balance {target_name}")
                X_res, y_res = SMOTE(random_state=self.random_state).fit_resample(X, y)
                return X_res, y_res
        return X, y

    def train_for_target(self, target_name):
        """Complete training pipeline"""
        if not self.selected_models:
            raise ValueError("No models selected. Use select_models() first")
            
        print(f"\n=== Training for {target_name} ===")
        print(f"Class distribution:\n{self.df[target_name].value_counts()}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(target_name)
        
        # Preprocess features FIRST
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Handle imbalance AFTER preprocessing
        X_train_processed, y_train = self._handle_imbalance(X_train_processed, y_train, target_name)
        
        # Train and evaluate selected models
        results = {}
        for model_name in self.selected_models:
            print(f"\nTraining {model_name}...")
            model = clone(self.available_models[model_name])
            
            # Special handling for ordinal targets
            if target_name in ['Physical Activity', 'Income Level'] and model_name == "Gradient Boosting":
                model.set_params(loss='exponential')
            
            model.fit(X_train_processed, y_train)
            y_pred = model.predict(X_test_processed)
            
            results[model_name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Confusion Matrix": confusion_matrix(y_test, y_pred),
                "Classification Report": classification_report(y_test, y_pred),
            }

            # Add visualization for Decision Tree
            if model_name == "Decision Tree":
                from sklearn.tree import plot_tree
                plt.figure(figsize=(20,10))
                plot_tree(model, filled=True, feature_names=self.preprocessor.get_feature_names_out(), 
                            class_names=[str(c) for c in model.classes_], max_depth=3)
                plt.title(f"Decision Tree for {target_name}")
                plt.show()
        
        return results