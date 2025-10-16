import os
import librosa
import numpy as np

# FOLDER DIRECTORY 
input_dir = r'/Users/nataliamorenoblasco/Desktop/AffectiveComputing_SpeechRecognition/Affective-Computing---Speech-Emotion-Recognition/archive-5'  

os.makedirs(output_dir, exist_ok=True)

# AUDIO & MFCC PARAMETERS 
SAMPLE_RATE = 48000    
N_MFCC = 13            
N_FFT = 2048           
HOP_LENGTH = 512      
FIXED_FRAMES = 128     

# MFCC GENERATION 
def process_file(input_path):
    y, sr = librosa.load(input_path, sr=SAMPLE_RATE)

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Pad to fixed frame length
    if mfccs.shape[1] < FIXED_FRAMES:
        mfccs = np.pad(mfccs, ((0,0), (0, FIXED_FRAMES - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :FIXED_FRAMES]

    return mfccs

#  PROCESS ALL ACTORS 
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

        # Save MFCC matrix as .npy
        output_path = os.path.join(actor_output_path, file.replace('.wav', '.npy'))
        np.save(output_path, mfcc_matrix)

        print(f" Saved: {output_path}")

print(" Done processing all actors!")
