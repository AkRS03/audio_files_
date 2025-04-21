import requests
from fastapi import FastAPI, status, HTTPException
from requests.auth import HTTPBasicAuth
import re
import os
from pydantic import BaseModel
from services.diarizing import diarize_with_speechbrain, diarize_speakers

class DownloadRequest(BaseModel):
    url: str
    username: str
    password: str
    order_id: str
    SRN_id: str

app=FastAPI()
headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'}

# url='https://recordings.exotel.com/exotelrecordings/readyassist/a8dcf42fe9f0efd1c76398292fed1941.mp3'
# username="6715743c745a1faf8fdbae5d27b35688035aab94bffc6590"
# passowrd="dadd54a50560718ef10a363d3d58ff6463150a7e9a4f3467"
# class DownloadRequest(BaseModel):
#     url: str
#     username: str
#     password: str
#     order_id: str
#     SRN_id: str

import librosa
import numpy as np

@app.post('/audio', status_code=status.HTTP_200_OK)
def post_audio(request: DownloadRequest):
    global headers
    response = requests.get(request.url, headers=headers, auth=HTTPBasicAuth(username=request.username, password=request.password), allow_redirects=True)

    file_name = re.split('readyassist/', str(request.url))[1]
    with open(file_name, 'wb') as file:
        file.write(response.content)
        file_size = file.tell() / 1024  
    y, sr = librosa.load(file_name, sr=None) 
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
   
    return {
        "file_name": file_name,
        "Username": request.username,
        "order_id": request.order_id,
        "SRN_id": request.SRN_id,
        "File_size": round(file_size, 2),
        "Duration": round(librosa.get_duration(y=y, sr=sr), 2),
        "Sampling_Rate": sr,
        "Spectral_Centroid": float(np.mean(spectral_centroid)),
        "Spectral_Bandwidth": float(np.mean(spectral_bandwidth))
        }
@app.get('/analyze_speaker', status_code=200)
def analyze_speaker(url: str):
    """Detect speaker changes and predict gender based on pitch."""

    file_name = r'C:\Users\Lenovo\OneDrive\Documents\audio_scraping\a8dcf42fe9f0efd1c76398292fed1941.mp3'

    # Load audio
    y, sr = librosa.load(file_name, sr=None)
    frame_length = 2048
    hop_length = 512
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # Compute energy differences (spikes indicate speaker change)
    energy_diff = np.abs(np.diff(energy))

    # Detect significant changes
    change_points = np.where(energy_diff > np.percentile(energy_diff, 90))[0]  

    pitches = librosa.piptrack(y=y, sr=sr)[0]
    pitches = pitches[pitches > 0]  # Remove non-pitched values
    avg_pitch = np.mean(pitches) if len(pitches) > 0 else 0

    # Gender classification
    gender = "Unknown"
    if avg_pitch > 165:  
        gender = "Female"
    elif avg_pitch > 85:
        gender = "Male"

    return {
    "file_name": file_name,
    "sampling_rate": int(sr),  # convert from numpy.int32 to int
    "duration_sec": round(librosa.get_duration(y=y, sr=sr), 2),
    "speaker_change_points": change_points.tolist(),  # already native list
    "avg_pitch_Hz": round(float(avg_pitch), 2),  # ensure float, not numpy.float32
    "predicted_gender": gender
}
@app.post('/split_and_analyze', status_code=200)
def split_and_analyze(url: str):
    """Split audio by speaker changes, classify gender, and calculate speaking rate."""
    file_name = re.split('readyassist/', str(url))[1]

    # Load audio
    y, sr = librosa.load(file_name, sr=None)
    sr = int(sr)  # Ensure it's a native int
    frame_length = 2048
    hop_length = 512
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Compute energy differences (spikes indicate speaker change)
    energy_diff = np.abs(np.diff(energy))
    change_points = np.where(energy_diff > np.percentile(energy_diff, 90))[0] * hop_length
    change_points = np.concatenate(([0], change_points, [len(y)]))

    segments = []
    for i in range(len(change_points) - 1):
        start = int(change_points[i])
        end = int(change_points[i + 1])
        segment = y[start:end]
        duration = float(librosa.get_duration(y=segment, sr=sr))

        # Analyze pitch for gender classification
        pitches, _ = librosa.piptrack(y=segment, sr=sr)
        pitches = pitches[pitches > 0]
        avg_pitch = float(np.mean(pitches)) if len(pitches) > 0 else 0.0

        gender = "Unknown"
        if avg_pitch > 165:
            gender = "Female"
        elif avg_pitch > 85:
            gender = "Male"

        # Speaking rate
        zero_crossings = np.sum(librosa.zero_crossings(segment, pad=False))
        speaking_rate = zero_crossings / duration if duration > 0 else 0

        segments.append({
            "start_time_sec": round(start / sr, 2),
            "end_time_sec": round(end / sr, 2),
            "duration_sec": round(duration, 2),
            "avg_pitch_Hz": round(avg_pitch, 2),
            "predicted_gender": gender,
            "speaking_rate_words_per_sec": round(speaking_rate, 2)
        })

    return {
        "file_name": file_name,
        "sampling_rate": sr,
        "total_duration_sec": round(float(librosa.get_duration(y=y, sr=sr)), 2),
        "segments": segments
    }

@app.get('/conversation_audio', status_code=200)
def get_conversation_details(file_path: str):
    return diarize_speakers(file_path)