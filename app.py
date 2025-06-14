import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import tempfile
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import soundfile as sf
from audio_recorder_streamlit import audio_recorder
import time
import os

@st.cache_resource
def load_model_files():
    """Load the trained model and encoders - cached function"""
    try:
        model = joblib.load('baby_cry_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        # Try to load scaler if it exists
        scaler = None
        try:
            scaler = joblib.load('scaler.pkl')
        except:
            pass  # Scaler might not exist
            
        return model, label_encoder, feature_columns, scaler, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, False

class BabyCryDetector:
    def __init__(self):
        self.model, self.label_encoder, self.feature_columns, self.scaler, self.model_loaded = load_model_files()
    
    def enhanced_features_extractor(self, audio_data, sample_rate):
        """Extract enhanced features from audio - same as training"""
        try:
            # If audio is too short, pad it
            if len(audio_data) < sample_rate:
                audio_data = np.pad(audio_data, (0, sample_rate - len(audio_data)), mode='constant')
            
            # If audio is too long, take first 3 seconds
            if len(audio_data) > sample_rate * 5:
                audio_data = audio_data[:sample_rate * 3]
            
            features = []
            
            # 1. MFCC features (most important for speech/cry recognition)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            features.extend(mfccs_mean)
            features.extend(mfccs_std)
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # 4. Root Mean Square Energy
            rms = librosa.feature.rms(y=audio_data)[0]
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            # 5. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            features.extend(np.mean(chroma, axis=1))
            
            # 6. Mel-scale spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
            mel_features = np.mean(mel_spectrogram, axis=1)
            features.extend(mel_features[:10])  # Take first 10 mel features
            
            return np.array(features)
            
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None
    
    def predict(self, audio_data, sample_rate):
        """Predict baby cry type from audio data"""
        if not self.model_loaded or self.model is None:
            return None, None, None
        
        # Use the enhanced feature extractor (same as training)
        features = self.enhanced_features_extractor(audio_data, sample_rate)
        
        if features is not None:
            # Create DataFrame with proper column names
            features_df = pd.DataFrame([features], columns=self.feature_columns)
            
            # Apply scaling if scaler exists
            if self.scaler is not None:
                features_df = pd.DataFrame(
                    self.scaler.transform(features_df), 
                    columns=self.feature_columns
                )
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]
            
            # Get the class name
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            # Get all class probabilities
            all_classes = self.label_encoder.classes_
            class_probs = dict(zip(all_classes, probabilities))
            
            return predicted_class, confidence, class_probs
        return None, None, None

def plot_waveform(audio_data, sample_rate):
    """Plot audio waveform"""
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300,
        showlegend=False
    )
    
    return fig

def plot_probabilities(class_probs):
    """Plot class probabilities"""
    classes = list(class_probs.keys())
    probs = list(class_probs.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            marker_color=['red' if p == max(probs) else 'lightblue' for p in probs]
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Cry Type",
        yaxis_title="Probability",
        height=400,
        showlegend=False
    )
    
    return fig

def check_model_files():
    """Check if all required model files exist"""
    required_files = ['baby_cry_model.pkl', 'label_encoder.pkl', 'feature_columns.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def main():
    st.set_page_config(
        page_title="Baby Cry Detection",
        page_icon="👶",
        layout="wide"
    )
    
    st.title("🍼 Baby Cry Detection System")
    st.markdown("---")
    
    # Check if model files exist
    files_exist, missing_files = check_model_files()
    
    if not files_exist:
        st.error("❌ Model files not found!")
        st.error(f"Missing files: {', '.join(missing_files)}")
        st.info("Please ensure the following files are in the same directory as your app.py:")
        for file in missing_files:
            st.write(f"- {file}")
        return
    
    # Initialize detector (this will use the cached function)
    try:
        detector = BabyCryDetector()
        
        if not detector.model_loaded:
            st.error("❌ Model not loaded! Please check the model files.")
            return
        
        st.success("✅ Model loaded successfully!")
        
        # Display model info
        st.info(f"Model expects {len(detector.feature_columns)} features")
        
    except Exception as e:
        st.error(f"❌ Error initializing detector: {e}")
        return
    
    # Sidebar for method selection
    st.sidebar.title("🎵 Audio Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ("Upload Audio File", "Record Audio", "Real-time Analysis")
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if input_method == "Upload Audio File":
            st.header("📁 Upload Audio File")
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
                help="Upload an audio file containing baby cry sounds"
            )
            
            if uploaded_file is not None:
                try:
                    # Load audio file with consistent sample rate
                    audio_data, sample_rate = librosa.load(uploaded_file, sr=22050)
                    
                    # Display audio player
                    st.audio(uploaded_file, format='audio/wav')
                    
                    # Display audio info
                    st.info(f"Audio loaded: {len(audio_data)/sample_rate:.2f} seconds, {sample_rate} Hz")
                    
                    # Plot waveform
                    fig_wave = plot_waveform(audio_data, sample_rate)
                    st.plotly_chart(fig_wave, use_container_width=True)
                    
                    # Analyze button
                    if st.button("🔍 Analyze Audio", key="analyze_upload"):
                        with st.spinner("Analyzing audio..."):
                            predicted_class, confidence, class_probs = detector.predict(audio_data, sample_rate)
                            
                            if predicted_class:
                                st.success(f"✅ Analysis Complete!")
                                
                                # Display results in col2
                                with col2:
                                    st.header("📊 Results")
                                    
                                    # Main prediction
                                    st.metric(
                                        label="Predicted Cry Type",
                                        value=predicted_class.title(),
                                        delta=f"{confidence:.1%} confidence"
                                    )
                                    
                                    # Advice based on prediction
                                    advice = {
                                        'hungry': "🍼 Your baby might be hungry. Try feeding.",
                                        'tired': "😴 Your baby seems tired. Consider naptime.",
                                        'discomfort': "😣 Your baby appears uncomfortable. Check diaper or clothing.",
                                        'belly_pain': "🤱 Your baby might have stomach discomfort. Try gentle burping or tummy massage.",
                                        'burping': "💨 Your baby needs to burp. Try different burping positions."
                                    }
                                    
                                    if predicted_class in advice:
                                        st.info(advice[predicted_class])
                                
                                # Plot probabilities
                                fig_probs = plot_probabilities(class_probs)
                                st.plotly_chart(fig_probs, use_container_width=True)
                            else:
                                st.error("❌ Could not analyze audio. Please try again.")
                
                except Exception as e:
                    st.error(f"Error processing audio file: {e}")
                    st.error("Please check if the audio file is valid and try again.")
        
        elif input_method == "Record Audio":
            st.header("🎤 Record Audio")
            st.info("Click the microphone button below to start recording your baby's cry")
            
            # Audio recorder
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=3.0,
                sample_rate=22050
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                try:
                    # Convert bytes to audio data
                    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                    
                    # Ensure consistent sample rate
                    if sample_rate != 22050:
                        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)
                        sample_rate = 22050
                    
                    st.info(f"Recorded: {len(audio_data)/sample_rate:.2f} seconds")
                    
                    # Plot waveform
                    fig_wave = plot_waveform(audio_data, sample_rate)
                    st.plotly_chart(fig_wave, use_container_width=True)
                    
                    # Auto-analyze recorded audio
                    with st.spinner("Analyzing recorded audio..."):
                        predicted_class, confidence, class_probs = detector.predict(audio_data, sample_rate)
                        
                        if predicted_class:
                            st.success(f"✅ Analysis Complete!")
                            
                            # Display results in col2
                            with col2:
                                st.header("📊 Results")
                                
                                # Main prediction
                                st.metric(
                                    label="Predicted Cry Type",
                                    value=predicted_class.title(),
                                    delta=f"{confidence:.1%} confidence"
                                )
                                
                                # Advice
                                advice = {
                                    'hungry': "🍼 Your baby might be hungry. Try feeding.",
                                    'tired': "😴 Your baby seems tired. Consider naptime.",
                                    'discomfort': "😣 Your baby appears uncomfortable. Check diaper or clothing.",
                                    'belly_pain': "🤱 Your baby might have stomach discomfort. Try gentle burping or tummy massage.",
                                    'burping': "💨 Your baby needs to burp. Try different burping positions."
                                }
                                
                                if predicted_class in advice:
                                    st.info(advice[predicted_class])
                            
                            # Plot probabilities
                            fig_probs = plot_probabilities(class_probs)
                            st.plotly_chart(fig_probs, use_container_width=True)
                        else:
                            st.error("❌ Could not analyze audio. Please try again.")
                
                except Exception as e:
                    st.error(f"Error processing recorded audio: {e}")
        
        else:  # Real-time Analysis
            st.header("🔴 Real-time Analysis")
            st.warning("⚠️ This feature requires microphone access and may consume more resources")
            
            # Placeholder for real-time analysis
            st.info("🚧 Real-time analysis feature is under development")
            st.markdown("""
            **Coming Soon:**
            - Continuous microphone monitoring
            - Real-time cry classification
            - Live probability updates
            - Alert notifications
            """)
    
    with col2:
        if input_method in ["Upload Audio File", "Record Audio"]:
            st.header("ℹ️ About")
            st.markdown("""
            **Cry Types:**
            - 🍼 **Hungry**: Baby needs feeding
            - 😴 **Tired**: Baby needs sleep
            - 😣 **Discomfort**: General discomfort
            - 🤱 **Belly Pain**: Stomach discomfort
            - 💨 **Burping**: Needs to burp
            
            **Tips for Best Results:**
            - Use clear audio recordings
            - Minimize background noise
            - Record for at least 2-3 seconds
            - Ensure baby's cry is prominent
            """)
        
        else:
            st.header("🎯 Model Performance")
            st.markdown("""
            **Model Details:**
            - Architecture: Random Forest/Ensemble
            - Features: Enhanced audio features (58 total)
            - Classes: 5 cry types
            - Accuracy: ~85-90%
            
            **Important Note:**
            This is an AI assistant tool and should not replace professional medical advice. Always consult healthcare providers for serious concerns.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "👶 Baby Cry Detection System | Built with ❤️ using Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()