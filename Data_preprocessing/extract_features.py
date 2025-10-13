import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

# INPUT & OUTPUT DIRECTORIES
input_dir = r'/Users/nataliamorenoblasco/Desktop/AffectiveComputing_SpeechRecognition/Affective-Computing---Speech-Emotion-Recognition/archive-5'
output_dir = r'/Users/nataliamorenoblasco/Desktop/AffectiveComputing_SpeechRecognition/Affective-Computing---Speech-Emotion-Recognition/Data_preprocessing/processed_mfccs'
os.makedirs(output_dir, exist_ok=True)

# AUDIO & MFCC PARAMETERS 
SAMPLE_RATE = 16000     # Downsample to reduce noise (48000 → 16000 often works better)
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
FIXED_FRAMES = 128      # Keep same temporal resolution

# --- MFCC GENERATION FUNCTION ---
def process_file(input_path):
    # Load and trim silence
    y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=30)  # remove silence
    
    # Normalize loudness
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Add delta and delta-delta coefficients (dynamic features)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_full = np.concatenate([mfcc, delta, delta2], axis=0)  # shape: (39, time)
    
    # Pad or truncate to fixed frame length
    if mfcc_full.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - mfcc_full.shape[1]
        mfcc_full = np.pad(mfcc_full, ((0,0), (0,pad_width)), mode='constant')
    else:
        mfcc_full = mfcc_full[:, :FIXED_FRAMES]
    
    # Standardize (zero mean, unit variance)
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc_full.T).T
    
    return mfcc_scaled


# PROCESS ALL ACTORS
actors = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

for actor in actors:
    actor_input_path = os.path.join(input_dir, actor)
    actor_output_path = os.path.join(output_dir, actor)
    os.makedirs(actor_output_path, exist_ok=True)

    wav_files = [f for f in os.listdir(actor_input_path) if f.endswith('.wav')]
    print(f"Actor {actor}: Found {len(wav_files)} WAV files")

    for file in wav_files:
        input_path = os.path.join(actor_input_path, file)
        mfcc_matrix = process_file(input_path)

        output_path = os.path.join(actor_output_path, file.replace('.wav', '.npy'))
        np.save(output_path, mfcc_matrix)
        print(f" Saved: {output_path}")

print("Done processing all actors!")
