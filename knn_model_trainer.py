"""
KNN Model untuk Prediksi Risiko Stunting
Training dan deployment model untuk website
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

class StuntingKNNModel:
    """
    Model KNN untuk prediksi risiko stunting
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.gender_encoder = None
        self.feature_names = ['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']
        
    def load_and_prepare_data(self):
        """
        Load dan persiapan data dari CSV
        """
        print("📂 Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nData Info:")
        print(df.info())
        print(f"\nStatus Gizi Distribution:")
        print(df['Status Gizi'].value_counts())
        
        # Check missing values
        if df.isnull().sum().sum() > 0:
            print("\n⚠️ Missing values detected. Cleaning...")
            df = df.dropna()
        
        return df
    
    def encode_features(self, df):
        """
        Encode categorical features
        """
        # Encode Jenis Kelamin
        self.gender_encoder = LabelEncoder()
        df['Jenis Kelamin Encoded'] = self.gender_encoder.fit_transform(df['Jenis Kelamin'])
        
        # Encode Status Gizi (target)
        self.label_encoder = LabelEncoder()
        df['Status Gizi Encoded'] = self.label_encoder.fit_transform(df['Status Gizi'])
        
        print("\n🔄 Encoding completed:")
        print(f"Gender mapping: {dict(zip(self.gender_encoder.classes_, self.gender_encoder.transform(self.gender_encoder.classes_)))}")
        print(f"Status mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return df
    
    def train_model(self, optimize=True):
        """
        Training KNN model dengan optimasi hyperparameter
        """
        # Load data
        df = self.load_and_prepare_data()
        df = self.encode_features(df)
        
        # Prepare features and target
        X = df[['Umur (bulan)', 'Jenis Kelamin Encoded', 'Tinggi Badan (cm)']].values
        y = df['Status Gizi Encoded'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data split:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        if optimize:
            print("\n🔍 Optimizing hyperparameters...")
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
            
            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(
                knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"\n✅ Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Default parameters
            self.model = KNeighborsClassifier(n_neighbors=5, weights='distance')
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluation
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=self.label_encoder.classes_))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def save_model(self, model_dir):
        """
        Save model, scaler, dan encoders
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'knn_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save encoders
        encoders_path = os.path.join(model_dir, 'encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump({
                'gender_encoder': self.gender_encoder,
                'label_encoder': self.label_encoder
            }, f)
        
        print(f"\n💾 Model saved to {model_dir}")
        print(f"   - knn_model.pkl")
        print(f"   - scaler.pkl")
        print(f"   - encoders.pkl")
    
    def load_model(self, model_dir):
        """
        Load saved model
        """
        model_path = os.path.join(model_dir, 'knn_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        encoders_path = os.path.join(model_dir, 'encoders.pkl')
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
            self.gender_encoder = encoders['gender_encoder']
            self.label_encoder = encoders['label_encoder']
        
        print("✅ Model loaded successfully")
    
    def predict(self, age_months, gender, height_cm):
        """
        Prediksi status gizi untuk input baru
        
        Returns:
        - prediction: Status gizi prediksi
        - probability: Probabilitas untuk setiap kelas
        - risk_percentage: Persentase risiko stunting
        """
        # Normalize gender to lowercase to match dataset format
        gender = gender.lower()
        
        # Encode gender
        gender_encoded = self.gender_encoder.transform([gender])[0]
        
        # Prepare input
        X_input = np.array([[age_months, gender_encoded, height_cm]])
        X_input_scaled = self.scaler.transform(X_input)
        
        # Predict
        prediction_encoded = self.model.predict(X_input_scaled)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_input_scaled)[0]
        prob_dict = dict(zip(self.label_encoder.classes_, probabilities))
        
        # Calculate stunting risk (severely stunted + stunted)
        risk_classes = ['severely stunted', 'stunted']
        risk_percentage = sum([prob_dict.get(cls, 0) for cls in risk_classes]) * 100
        
        return {
            'prediction': prediction,
            'probabilities': prob_dict,
            'risk_percentage': round(risk_percentage, 2)
        }
    
    def get_risk_interpretation(self, risk_percentage):
        """
        Interpretasi persentase risiko
        """
        if risk_percentage >= 70:
            return {
                'level': 'SANGAT TINGGI',
                'color': '#DC2626',
                'message': 'Berdasarkan data training, karakteristik anak ini sangat mirip dengan anak-anak yang mengalami stunting dalam dataset.'
            }
        elif risk_percentage >= 50:
            return {
                'level': 'TINGGI',
                'color': '#F59E0B',
                'message': 'Berdasarkan data training, anak ini memiliki karakteristik yang cukup mirip dengan anak-anak stunting dalam dataset.'
            }
        elif risk_percentage >= 30:
            return {
                'level': 'SEDANG',
                'color': '#F59E0B',
                'message': 'Anak ini memiliki beberapa karakteristik yang perlu diperhatikan, namun masih dalam batas wajar.'
            }
        else:
            return {
                'level': 'RENDAH',
                'color': '#10B981',
                'message': 'Berdasarkan data training, karakteristik anak ini mirip dengan anak-anak yang memiliki pertumbuhan normal dalam dataset.'
            }


# Main execution
if __name__ == "__main__":
    # Path configuration
    DATA_PATH = r"C:\Users\muham\Project\Stunting\data_balita.csv"
    MODEL_DIR = r"C:\Users\muham\Project\Stunting\models"
    
    # Initialize and train model
    print("=" * 80)
    print("🚀 TRAINING KNN MODEL UNTUK DETEKSI STUNTING")
    print("=" * 80)
    
    knn_model = StuntingKNNModel(DATA_PATH)
    
    # Train with optimization
    metrics = knn_model.train_model(optimize=True)
    
    # Save model
    knn_model.save_model(MODEL_DIR)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED")
    print("=" * 80)
    
    # Test prediction
    print("\n📝 Testing prediction...")
    result = knn_model.predict(24, 'laki-laki', 80)
    print(f"\nTest Input: Umur=24 bulan, Gender=laki-laki, TB=80cm")
    print(f"Prediction: {result['prediction']}")
    print(f"Risk Percentage: {result['risk_percentage']}%")
    print(f"Probabilities: {result['probabilities']}")