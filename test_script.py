import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import sys

# === Load model and encoder ===
model_path = "emotion_model_filtered_final.h5"
encoder_path = "label_encoder.pkl"

model = load_model(model_path)
le = joblib.load(encoder_path)

# === Prediction function ===
def predict_emotion(audio_path):
    try:
        audio, sr = librosa.load(audio_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        if mfcc.shape[1] < 130:
            mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :130]

        mfcc = mfcc[np.newaxis, ..., np.newaxis]

        prediction = model.predict(mfcc)
        emotion_idx = np.argmax(prediction)
        emotion = le.inverse_transform([emotion_idx])[0]

        print(f"🎧 Predicted Emotion: {emotion}")
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_script.py <path_to_audio.wav>")
    else:
        predict_emotion(sys.argv[1])