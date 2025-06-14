# 👶 Baby Cry Detection System

A machine learning-powered web application that analyzes baby cries to determine the likely cause (hungry, tired, discomfort, belly pain, or burping). Built with Random Forest classifier using enhanced audio features and deployed with Streamlit.

## 🎯 Features

- **Multiple Input Methods**: Upload audio files or record live audio through the web interface
- **5 Cry Classifications**: Hungry, Tired, Discomfort, Belly Pain, Burping
- **Real-time Visualization**: Audio waveforms and prediction probabilities
- **User-friendly Interface**: Intuitive Streamlit web application
- **Actionable Insights**: Provides caring advice based on predictions
- **Enhanced Feature Extraction**: 58 audio features including MFCC, spectral features, and more

## 🏗️ Model Architecture

- **Algorithm**: Random Forest Classifier (best performing among tested models)
- **Features**: 58 enhanced audio features including:
  - MFCC (Mel-Frequency Cepstral Coefficients) - 26 features
  - Spectral features (centroids, rolloff, bandwidth) - 6 features
  - Zero crossing rate - 2 features
  - Root Mean Square Energy - 2 features
  - Chroma features - 12 features
  - Mel-scale spectrogram - 10 features
- **Training Environment**: Google Colab with GPU acceleration
- **Performance**: ~85-90% accuracy on test set
- **Data Balancing**: SMOTE (Synthetic Minority Oversampling Technique)

## 📊 Dataset

The model is trained on audio files organized in the following categories:
- **Belly Pain**: Cries indicating stomach discomfort
- **Burping**: Cries indicating need to burp
- **Discomfort**: General discomfort cries
- **Hungry**: Cries indicating hunger
- **Tired**: Cries indicating tiredness/sleepiness

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Microphone access (for recording features)
- Web browser

### Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/baby-cry-detection.git
cd baby-cry-detection
```

2. **Create virtual environment**
```bash
python -m venv baby_cry_env
source baby_cry_env/bin/activate  # On Windows: baby_cry_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Ensure model files are present**
Make sure these files are in your project directory:
- `baby_cry_model.pkl` (trained Random Forest model)
- `label_encoder.pkl` (label encoder for cry types)
- `feature_columns.pkl` (feature column names)
- `scaler.pkl` (feature scaler)

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open your browser** to `http://localhost:8501`

## 🔧 Model Training (Google Colab)

The model training was performed in Google Colab with the following process:

### Training Pipeline

1. **Data Preparation**
   - Mount Google Drive containing the dataset
   - Organize audio files in category folders
   - Extract enhanced audio features from each file

2. **Feature Engineering**
   ```python
   # 58 total features extracted:
   # - 13 MFCC coefficients (mean + std) = 26 features
   # - Spectral features (centroid, rolloff, bandwidth) = 6 features  
   # - Zero crossing rate (mean + std) = 2 features
   # - RMS energy (mean + std) = 2 features
   # - 12 Chroma features
   # - 10 Mel-spectrogram features
   ```

3. **Model Training & Selection**
   - Tested multiple algorithms: Random Forest, Gradient Boosting, SVM
   - Applied SMOTE for class imbalance handling
   - Used cross-validation for robust evaluation
   - Selected best performing model (Random Forest)

4. **Model Export**
   - Saved trained model and preprocessing objects as pickle files
   - Downloaded from Colab to local project directory

### Training Code Structure (Colab)
```python
# Main training class
class ImprovedBabyCryDetectionTrainer:
    - enhanced_features_extractor()  # Extract 58 audio features
    - balance_dataset()              # Apply SMOTE balancing
    - train_multiple_models()        # Train and compare models
    - cross_validate_model()         # Perform cross-validation
    - evaluate_model()               # Comprehensive evaluation
    - save_model()                   # Export trained model
```

## 📁 Project Structure

```
baby-cry-detection/
├── app.py                      # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── baby_cry_model.pkl        # Trained Random Forest model
├── label_encoder.pkl         # Label encoder for cry categories
├── feature_columns.pkl       # Feature column names (58 features)
├── scaler.pkl               # Standard scaler for feature normalization
├── colab_training.ipynb     # Google Colab training notebook
└── data/                    # Dataset directory (for training)
    ├── belly_pain/
    ├── burping/
    ├── discomfort/  
    ├── hungry/
    └── tired/
```

## 🎨 Application Interface

### Main Features
- **Audio Upload**: Support for WAV, MP3, FLAC, M4A, OGG formats
- **Live Recording**: Record baby cries directly through the browser
- **Waveform Visualization**: Interactive audio waveform display
- **Probability Charts**: Confidence scores for all cry types
- **Caring Advice**: Specific suggestions based on predictions

### Usage Methods

1. **Upload Audio File**:
   - Select "Upload Audio File" from sidebar
   - Choose an audio file from your device
   - Click "Analyze Audio" to get predictions

2. **Record Audio**:
   - Select "Record Audio" from sidebar  
   - Click the microphone button to start recording
   - Audio is automatically analyzed after recording stops

3. **Real-time Analysis**:
   - Feature planned for future release
   - Will provide continuous monitoring capabilities

## 🔍 Technical Details

### Audio Processing
- **Supported Formats**: WAV, MP3, FLAC, M4A, OGG
- **Sample Rate**: 22050 Hz (automatically resampled)
- **Feature Extraction**: 58 enhanced audio features per sample
- **Audio Duration**: Handles 2-3 second clips optimally

### Model Specifications
```python
# Final Random Forest Configuration
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
```

### Feature Extraction Process
```python
def enhanced_features_extractor(audio_data, sample_rate):
    # 1. MFCC features (26 total: 13 mean + 13 std)
    # 2. Spectral features (6 total: centroid, rolloff, bandwidth)
    # 3. Zero crossing rate (2 total: mean + std)
    # 4. RMS energy (2 total: mean + std) 
    # 5. Chroma features (12 total)
    # 6. Mel-spectrogram features (10 total)
    return features_array  # Shape: (58,)
```

## 📈 Model Performance

### Training Results
- **Cross-validation F1-Score**: ~0.85-0.90
- **Test Accuracy**: ~85-90%
- **Class Balance**: SMOTE applied for equal representation
- **Feature Importance**: MFCC features most discriminative

### Prediction Output
- **Primary Prediction**: Most likely cry type
- **Confidence Score**: Probability of primary prediction
- **All Probabilities**: Scores for all 5 cry types
- **Caring Advice**: Contextual suggestions for parents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Future Enhancements

- [ ] Real-time continuous monitoring
- [ ] Mobile app version  
- [ ] Advanced deep learning models (CNN/RNN)
- [ ] Multi-language interface
- [ ] Parent dashboard with crying pattern analytics
- [ ] Cloud deployment (Heroku/Streamlit Cloud)
- [ ] Integration with baby monitoring devices

## ⚠️ Important Disclaimers

- This is an **AI assistant tool** and should **not replace professional medical advice**
- Always consult healthcare providers for serious concerns about your baby
- The model provides probabilistic predictions, not definitive diagnoses
- Accuracy may vary based on recording quality and environmental factors
- This tool is designed to **assist and support** parents, not replace parental intuition

## 🛠️ Troubleshooting

### Common Issues

1. **"Model files not found" error**:
   - Ensure all .pkl files are in the same directory as app.py
   - Check file names match exactly: `baby_cry_model.pkl`, `label_encoder.pkl`, etc.

2. **Audio recording not working**:
   - Grant microphone permissions in your browser
   - Try refreshing the page and allowing permissions again

3. **"Could not analyze audio" error**:
   - Ensure audio file is in supported format
   - Check that audio contains actual sound (not silent)
   - Try with a different audio file

4. **Installation errors**:
   - Update pip: `pip install --upgrade pip`
   - Install packages individually if requirements.txt fails
   - Check Python version compatibility (3.8+)

### Performance Tips

- **Recording Quality**: Use quiet environments with minimal background noise
- **Duration**: Record for 2-3 seconds to capture sufficient features
- **Microphone**: Position microphone close to baby for clear audio
- **File Format**: WAV files generally provide best results
- **Browser**: Chrome/Firefox recommended for audio recording features

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 1GB free space
- **CPU**: Dual-core processor
- **Internet**: For initial package installation
- **Browser**: Chrome, Firefox, Safari, Edge (latest versions)

### Recommended Requirements
- **RAM**: 8GB+
- **Storage**: 2GB+ free space
- **CPU**: Quad-core processor
- **Microphone**: Built-in or external microphone for recording

## 🔒 Privacy & Security

- **Local Processing**: All audio analysis happens on your device
- **No Data Storage**: Audio files are processed in memory and not saved
- **No Cloud Upload**: Your baby's audio never leaves your computer
- **Open Source**: Full code transparency for security review
- **Real-time Only**: No persistent storage of audio data

## 📚 Dependencies

### Core Libraries
```
streamlit>=1.28.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
plotly>=5.15.0
soundfile>=0.12.0
audio-recorder-streamlit>=0.0.8
```

## 🆘 Support & Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/baby-cry-detection/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/yourusername/baby-cry-detection/discussions)
- **Documentation**: This README and code comments
- **Community**: Join our discussions for tips and best practices

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Colab**: For providing free GPU training environment
- **Streamlit**: For the amazing web app framework
- **Librosa**: For comprehensive audio processing capabilities
- **Scikit-learn**: For machine learning algorithms and tools
- **Audio Processing Community**: For research and best practices
- **Parent Testers**: For feedback and real-world validation

## 🧪 Testing Your Setup

### Quick Test
1. Run the app: `streamlit run app.py`
2. Upload a sample audio file or record a short clip
3. Verify that predictions appear with confidence scores
4. Check that visualizations (waveform, probabilities) display correctly

### Model Validation
- The app will show "✅ Model loaded successfully!" if all files are found
- Feature count should show 58 features in the interface
- All 5 cry types should appear in probability charts

---

**Built with ❤️ for parents everywhere**

*This tool is designed to assist and support parents in understanding their baby's needs. Always trust your parental instincts and consult healthcare professionals when needed.*