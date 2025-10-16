import os
import numpy as np
import librosa

# This code was extracted from: https://www.kaggle.com/code/kayodeowoseni/ser-cnn-lpc-mfcc
# --- MFCC + LPC extraction for a single file ---
def process_file(input_path, n_mfcc=40, lpc_order=16, frame_length=512, hop_length=256, max_len=216, sr=16000):
    # Load and normalize
    y, _ = librosa.load(input_path, sr=sr)
    y = librosa.util.normalize(y)

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.fix_length(mfcc, size=max_len, axis=1)

    # --- LPC ---
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    coeffs = []
    for i in range(frames.shape[1]):
        try:
            a = librosa.lpc(frames[:, i], lpc_order)
            coeffs.append(a[1:])  # skip a0
        except:
            coeffs.append(np.zeros(lpc_order))
    lpc = np.array(coeffs).T
    lpc = librosa.util.fix_length(lpc, size=max_len, axis=1)

    # Combine MFCC + LPC
    combined = np.vstack([mfcc, lpc])  # shape: (56, 216)
    return combined

# --- PROCESS ALL ACTORS ---
input_dir = "/Users/nataliamorenoblasco/Desktop/AffectiveComputing_SpeechRecognition/Affective-Computing---Speech-Emotion-Recognition/archive-5"
output_dir = "/Users/nataliamorenoblasco/Desktop/AffectiveComputing_SpeechRecognition/Affective-Computing---Speech-Emotion-Recognition/Data_preprocessing/preprocees_mfccsLPC"
os.makedirs(output_dir, exist_ok=True)

actors = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

for actor in actors:
    actor_input_path = os.path.join(input_dir, actor)
    actor_output_path = os.path.join(output_dir, actor)
    os.makedirs(actor_output_path, exist_ok=True)

    wav_files = [f for f in os.listdir(actor_input_path) if f.endswith('.wav')]
    print(f"🎙️ Actor {actor}: Found {len(wav_files)} WAV files")

    for file in wav_files:
        input_path = os.path.join(actor_input_path, file)
        try:
            features = process_file(input_path)
            output_path = os.path.join(actor_output_path, file.replace('.wav', '.npy'))
            np.save(output_path, features)
            print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"⚠️ Skipped {file}: {e}")

print("🏁 Done processing all actors!")
