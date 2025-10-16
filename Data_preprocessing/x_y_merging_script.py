"""
import os
import numpy as np

#  Paths 
data_dir = r"/Users/nataliamorenoblasco/Desktop/AffectiveComputing_SpeechRecognition/Affective-Computing---Speech-Emotion-Recognition/Data_preprocessing/processed_mfccs"
actors = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

# Emotion mapping 
emotion_map = {
    '01': 0,  # Neutral
    '02': 1,  # Calm
    '03': 2,  # Happy
    '04': 3,  # Sad
    '05': 4,  # Angry
    '06': 5,  # Fearful
    '07': 6,  # Disgust
    '08': 7,  # Surprised
}

X_list = []
y_list = []

for actor in actors:
    actor_path = os.path.join(data_dir, actor)
    files = [f for f in os.listdir(actor_path) if f.endswith('.npy')]

    for file in files:
       
        try:
            emotion_code = file.split('-')[2]
            label = emotion_map.get(emotion_code)
            if label is None:
                continue  # skip unknown emotions

            mfcc = np.load(os.path.join(actor_path, file))
            X_list.append(mfcc)
            y_list.append(label)
        except Exception as e:
            print(f"Skipping {file}: {e}")

#  Convert to NumPy arrays 
X = np.array(X_list)  # shape: (num_samples, n_mfcc, frames)
y = np.array(y_list)  # shape: (num_samples,)

print("Merged dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)
np.save("X_merged_new.npy", X)
np.save("y_merged_new.npy", y)
#print("Saved X_merged.npy and y_merged.npy")
print("Saved X_merged_new.npy and y_merged_new.npy")
"""
#Iniyans code to run the code
import os
import numpy as np
import librosa
from sklearn.utils import shuffle

# Paths
data_dir = r"/Users/nataliamorenoblasco/Desktop/AffectiveComputing_SpeechRecognition/Affective-Computing---Speech-Emotion-Recognition/Data_preprocessing/preprocees_mfccsLPC"
actors = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

# Emotion mapping
emotion_map = {
    '01': 0, '02': 1, '03': 2, '04': 3,
    '05': 4, '06': 5, '07': 6, '08': 7,
}

X_list, y_list = [], []
expected_shape = (56, 216)  # (MFCC+LPC)

for actor in actors:
    actor_path = os.path.join(data_dir, actor)
    files = sorted([f for f in os.listdir(actor_path) if f.endswith('.npy')])
    print(f"Processing {actor}: {len(files)} files")

    for file in files:
        try:
            emotion_code = file.split('-')[2]
            label = emotion_map.get(emotion_code)
            if label is None:
                continue

            mfcc = np.load(os.path.join(actor_path, file))

            # Fix shape if needed
            if mfcc.shape != expected_shape:
                mfcc = librosa.util.fix_length(mfcc, size=expected_shape[1], axis=1)
                if mfcc.shape[0] != expected_shape[0]:
                    print(f"⚠️ Skipping {file}: unexpected feature dimension {mfcc.shape}")
                    continue

            X_list.append(mfcc)
            y_list.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

# Convert to arrays and shuffle
X = np.array(X_list)
y = np.array(y_list)
X, y = shuffle(X, y, random_state=42)

print("✅ Merged dataset created")
print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("X_merged_new.npy", X)
np.save("y_merged_new.npy", y)
print("💾 Saved X_merged_new.npy and y_merged_new.npy")
