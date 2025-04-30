import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
from datetime import datetime

class IntersectionRiskPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Prepare features for training or prediction"""
        # Ensure date is datetime
        df['crash_date'] = pd.to_datetime(df['crash_date'])
        
        # Add temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Add weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24],
                                  labels=['night', 'morning', 'afternoon', 'evening'],
                                  include_lowest=True)
        
        return df
    
    def train(self, df, target_column='had_crash'):
        """Train the model"""
        # Prepare features
        df = self.prepare_features(df)
        
        # Define feature columns
        categorical_features = ['nearest_intersection_id', 'time_of_day']
        numerical_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                            'month_sin', 'month_cos', 'is_weekend']
        weather_features = [col for col in df.columns if col.startswith('weather_')]
        
        all_features = categorical_features + numerical_features + weather_features
        self.feature_columns = all_features
        
        # Split data
        X = df[all_features]
        y = df[target_column]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', 'passthrough', numerical_features + weather_features)
            ]
        )
        
        # Create and train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        self.model = pipeline
        self.preprocessor = preprocessor
        
        # Evaluate
        train_score = pipeline.score(X_train, y_train)
        val_score = pipeline.score(X_val, y_val)
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Validation accuracy: {val_score:.4f}")
        
    def predict_risk(self, intersection_id, datetime_obj, weather_data):
        """Predict risk for a specific intersection at a given time"""
        # Create input DataFrame
        input_data = pd.DataFrame({
            'nearest_intersection_id': [intersection_id],
            'crash_date': [datetime_obj.date()],
            'hour': [datetime_obj.hour],
            'day_of_week': [datetime_obj.weekday()],
            'month': [datetime_obj.month]
        })
        
        # Add weather data
        for key, value in weather_data.items():
            input_data[f'weather_{key}'] = [value]
        
        # Prepare features
        input_data = self.prepare_features(input_data)
        
        # Predict probability
        risk_score = self.model.predict_proba(input_data[self.feature_columns])[0][1]
        
        return risk_score
    
    def save_model(self, path):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns
        }, path)
    
    def load_model(self, path):
        """Load a trained model"""
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.preprocessor = saved_data['preprocessor']
        self.feature_columns = saved_data['feature_columns']

# Usage example:
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('../data/processed/balanced_crash_dataset.csv')
    
    # Initialize and train model
    predictor = IntersectionRiskPredictor()
    predictor.train(df)
    
    # Example prediction
    current_time = datetime.now()
    weather_data = {
        'temperature': 72,
        'precipitation': 0,
        'wind_speed': 5,
        # ... other weather features
    }
    
    risk_score = predictor.predict_risk(
        intersection_id='intersection_123',
        datetime_obj=current_time,
        weather_data=weather_data
    )
    
    print(f"Risk score: {risk_score:.4f}")
    
    # Save model
    predictor.save_model('../models/intersection_risk_predictor.joblib') 