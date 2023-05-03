import streamlit as st
import numpy as np
import pickle
import librosa
import soundfile

def extract_feature(file_name, mfcc, chroma, mel):
  with soundfile.SoundFile(file_name) as sound_file:
    X = sound_file.read(dtype="float32")
    sample_rate=sound_file.samplerate
    if chroma:
      stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
      mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
      result=np.hstack((result, mfccs))
    if chroma:
      chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
      result=np.hstack((result, chroma))
    if mel:
      mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
      result=np.hstack((result, mel))
  return result

def predict_emotion(model, audio_file):
    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    return prediction[0]

# Load your trained model
model = pickle.load(open("emotion_recognition_model.pkl", "rb"))

st.title("Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Predicting emotion...")
    emotion = predict_emotion(model, "temp_audio.wav")
    st.write(f"The emotion in the speech is: {emotion}")