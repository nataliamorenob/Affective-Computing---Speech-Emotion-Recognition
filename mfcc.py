import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# Load metadata CSV
metadata = pd.read_csv("dataset.csv")

# Parameters
SR = 16000          # target sample rate
N_MFCC = 40         # number of MFCC coefficients
HOP_LENGTH = 512    # hop length in samples
N_FFT = 2048        # FFT window size

# Function to extract MFCC features from a single file
def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        # Aggregate: mean & std for each coefficient across time
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        # Concatenate into one feature vector
        features = np.concatenate([mfcc_mean, mfcc_std])
        return features
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# Process all files
feature_list = []
labels = []

print("Extracting MFCC features...")
for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    file_path = row["path"]
    emotion = row["emotion"]

    feats = extract_mfcc(file_path)
    if feats is not None:
        feature_list.append(feats)
        labels.append(emotion)

# Convert to DataFrame
X = np.array(feature_list)
y = np.array(labels)

# Create column names like mfcc1_mean ... mfcc40_std
cols_mean = [f"mfcc{i+1}_mean" for i in range(N_MFCC)]
cols_std = [f"mfcc{i+1}_std" for i in range(N_MFCC)]
feature_df = pd.DataFrame(X, columns=cols_mean + cols_std)
feature_df["emotion"] = y

# Save to CSV or NPZ
feature_df.to_csv("features_mfcc.csv", index=False)
np.savez("features_mfcc.npz", X=X, y=y)

print(f"✅ Saved {len(feature_df)} feature rows to 'features_mfcc.csv' and 'features_mfcc.npz'")
print(feature_df.head())
