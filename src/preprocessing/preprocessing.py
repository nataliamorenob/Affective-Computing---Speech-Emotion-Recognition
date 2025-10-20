import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

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


def process_file_mfcc(input_path):
    return process_file(input_path, n_mfcc=40, lpc_order=16, frame_length=512, hop_length=256, max_len=216, sr=16000)


def generate_features(input_dir, output_dir, feature_extraction="mfcc_lpc"):
    os.makedirs(output_dir, exist_ok=True)
    actors = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    for actor in actors:
        actor_input = os.path.join(input_dir, actor)
        actor_output = os.path.join(output_dir, actor)
        os.makedirs(actor_output, exist_ok=True)

        for file in [f for f in os.listdir(actor_input) if f.endswith(".wav")]:
            input_path = os.path.join(actor_input, file)
            output_path = os.path.join(actor_output, file.replace(".wav", ".npy"))

            if feature_extraction == "mfcc":
                features = process_file_mfcc(input_path)
            elif feature_extraction == "mfcc_lpc":
                features = process_file(input_path)
            else:
                raise ValueError(f"Unknown feature extraction method: {feature_extraction}")

            np.save(output_path, features)
            print(f"Saved {output_path}")

    print(f"Finished generating {feature_extraction.upper()} features.")


def merge_features(data_dir, output_path="", feature_extraction="mfcc_lpc"):
    emotion_map = {
        '01': 0, '02': 1, '03': 2, '04': 3,
        '05': 4, '06': 5, '07': 6, '08': 7,
    }

    X_list, y_list = [], []
    for actor in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor)
        if not os.path.isdir(actor_path):
            continue
        for file in os.listdir(actor_path):
            if not file.endswith(".npy"):
                continue
            emotion_code = file.split('-')[2]
            label = emotion_map.get(emotion_code)
            if label is None:
                continue
            arr = np.load(os.path.join(actor_path, file))
            X_list.append(arr)
            y_list.append(label)

    X, y = np.array(X_list), np.array(y_list)
    print(f"Merged dataset: X={X.shape}, y={y.shape}")

    np.save(os.path.join(output_path, f"X_merged_{feature_extraction}.npy"), X)
    np.save(os.path.join(output_path, f"y_merged_{feature_extraction}.npy"), y)
    print(f"Saved {feature_extraction.upper()} dataset to {output_path}")