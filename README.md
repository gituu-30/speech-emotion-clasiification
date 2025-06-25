# 🎙️ Speech Emotion Recognition App

This app classifies emotions from `.wav` speech using a trained CNN model based on MFCC features.

## ✅ Emotions Predicted:
- angry
- calm
- fearful
- happy
- neutral
- sad

## 🧠 Model Info
- Input: 40 MFCCs × 130 frames
- CNN model trained on RAVDESS (speech + song)
- Achieved ~80% macro F1 score

## 📁 Project Structure
speech_emotion_app/
├── app.py
├── emotion_model_filtered_final.h5
├── label_encoder.pkl
├── test_script.py
├── requirements.txt
├── README.m



## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```