import os
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import joblib
import warnings
warnings.filterwarnings('ignore')

class BabyCryDetectionTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        
    def traverse_subfolders(self, subfolders):
        """Traverse subfolders and get audio files with their labels"""
        audio_files = []
        subfolder_names = []

        for subfolder in subfolders:
            subfolder_name = os.path.basename(subfolder)
            files = self.get_audio_files(subfolder)
            audio_files.extend(files)
            subfolder_names.extend([subfolder_name] * len(files))

        return audio_files, subfolder_names

    def get_audio_files(self, subfolder):
        """Get all .wav files from a subfolder"""
        audio_files = []
        for root, dirs, files in os.walk(subfolder):
            for file in files:
                if file.endswith(".wav"):
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def features_extractor(self, file_name):
        """Extract MFCC features from audio file"""
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=30)
            # Extract MFCC features
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
            return mfccs_scaled_features
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            return None

    def process_audio_files(self, audio_files, subfolder_names):
        """Process all audio files and extract features"""
        data = []
        print("Processing audio files...")
        
        for i, (audio_file, subfolder_name) in enumerate(zip(audio_files, subfolder_names)):
            if i % 50 == 0:
                print(f"Processed {i}/{len(audio_files)} files")
                
            features = self.features_extractor(audio_file)
            if features is not None:
                data.append([audio_file, features, subfolder_name])

        df = pd.DataFrame(data, columns=["File", "Features", "Class"])
        return df

    def prepare_dataset(self):
        """Prepare the complete dataset"""
        # Define your subfolder paths here
        subfolders = [
            os.path.join(self.data_path, 'belly_pain'),
            os.path.join(self.data_path, 'burping'),
            os.path.join(self.data_path, 'discomfort'),
            os.path.join(self.data_path, 'hungry'),
            os.path.join(self.data_path, 'tired')
        ]
        
        # Get audio files and labels
        audio_files, subfolder_names = self.traverse_subfolders(subfolders)
        print(f"Found {len(audio_files)} audio files")
        
        # Process audio files
        df = self.process_audio_files(audio_files, subfolder_names)
        
        # Drop file column as we don't need it for training
        df = df.drop("File", axis=1)
        
        return df

    def visualize_data(self, df):
        """Visualize class distribution"""
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x="Class")
        plt.title("Distribution of Baby Cry Classes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def balance_dataset(self, X, y):
        """Balance the dataset using oversampling"""
        print("Balancing dataset...")
        resamp = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = resamp.fit_resample(X, y)
        
        print(f"Original dataset shape: {X.shape}")
        print(f"Resampled dataset shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled

    def prepare_features(self, df):
        """Convert features to proper format for training"""
        # Convert features to DataFrame
        features_df = pd.DataFrame(df['Features'].tolist())
        
        # Get labels
        y = df['Class']
        
        return features_df, y

    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        # Use Random Forest instead of Decision Tree for better performance
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def save_model(self, model_path="baby_cry_model.pkl", encoder_path="label_encoder.pkl"):
        """Save the trained model and label encoder"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        joblib.dump(self.feature_columns, "feature_columns.pkl")
        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")

    def predict_single_audio(self, audio_file_path):
        """Predict the class of a single audio file"""
        features = self.features_extractor(audio_file_path)
        if features is not None:
            features_df = pd.DataFrame([features], columns=self.feature_columns)
            prediction = self.model.predict(features_df)[0]
            probability = self.model.predict_proba(features_df)[0]
            
            # Get the class name
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probability)
            
            return predicted_class, confidence
        return None, None

    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting Baby Cry Detection Model Training...")
        
        # Prepare dataset
        df = self.prepare_dataset()
        print(f"Dataset prepared with {len(df)} samples")
        
        # Visualize data distribution
        self.visualize_data(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Store feature column names
        self.feature_columns = X.columns.tolist()
        
        # Balance dataset
        X_balanced, y_balanced = self.balance_dataset(X, y_encoded)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate model
        self.evaluate_model(X_test, y_test)
        
        # Save model
        self.save_model()
        
        print("Training pipeline completed successfully!")

# Usage example
if __name__ == "__main__":
    # Set your data path here
    DATA_PATH = "D:\Semester 07\FYP\Cry Detection Model Training\data"
    
    # Initialize trainer
    trainer = BabyCryDetectionTrainer(DATA_PATH)
    
    # Run training pipeline
    trainer.run_training_pipeline()
    
    # Example prediction
    # test_audio = "/path/to/test/audio.wav"
    # predicted_class, confidence = trainer.predict_single_audio(test_audio)
    # print(f"Predicted: {predicted_class} (Confidence: {confidence:.4f})")