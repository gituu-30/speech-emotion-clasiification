import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

# Load model and label encoder
@st.cache_resource
def load_artifacts():
    model = load_model("emotion_model_filtered_final.h5", compile=False)
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_artifacts()

# File upload
st.title("🎤 Speech Emotion Recognition")
uploaded_file = st.file_uploader("Upload a WAV file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    try:
        signal, sr = librosa.load(uploaded_file, sr=None, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < 130:
            mfcc = np.pad(mfcc, ((0,0),(0,130-mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :130]
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        pred = model.predict(mfcc)
        pred_label = encoder.inverse_transform([np.argmax(pred)])
        st.success(f"Predicted Emotion: {pred_label[0].capitalize()}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")