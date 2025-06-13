#!/usr/bin/env python3
"""
Simple training script for Baby Cry Detection
Run this script to train your model with minimal setup
"""

import os
import sys
from train_model import BabyCryDetectionTrainer

def main():
    print("🍼 Baby Cry Detection - Model Training")
    print("=" * 50)
    
    # Get data path from user
    data_path = input("Enter the path to your dataset folder: ").strip()
    
    # Validate path
    if not os.path.exists(data_path):
        print(f"❌ Error: Path '{data_path}' does not exist!")
        sys.exit(1)
    
    # Check if required subfolders exist
    required_folders = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    missing_folders = []
    
    for folder in required_folders:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"❌ Error: Missing required folders: {missing_folders}")
        print("Expected folder structure:")
        for folder in required_folders:
            print(f"  📁 {data_path}/{folder}/")
        sys.exit(1)
    
    print(f"✅ Dataset path validated: {data_path}")
    
    # Count audio files
    total_files = 0
    for folder in required_folders:
        folder_path = os.path.join(data_path, folder)
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        total_files += len(wav_files)
        print(f"  📊 {folder}: {len(wav_files)} files")
    
    print(f"📈 Total audio files: {total_files}")
    
    if total_files == 0:
        print("❌ Error: No .wav files found in the dataset!")
        sys.exit(1)
    
    # Confirm training
    response = input(f"\n🚀 Ready to train model with {total_files} files? (y/N): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    # Initialize trainer and run training
    try:
        print("\n🔄 Initializing trainer...")
        trainer = BabyCryDetectionTrainer(data_path)
        
        print("🎯 Starting training pipeline...")
        trainer.run_training_pipeline()
        
        print("\n🎉 Training completed successfully!")
        print("Generated files:")
        print("  📄 baby_cry_model.pkl")
        print("  📄 label_encoder.pkl") 
        print("  📄 feature_columns.pkl")
        
        print("\n🚀 You can now run the Streamlit app:")
        print("  streamlit run app.py")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()