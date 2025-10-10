import os
import pandas as pd

# Path to your dataset folder
base_dir = "data"

# Mappings
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

statement_map = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}

data = []

for actor_folder in sorted(os.listdir(base_dir)):
    actor_path = os.path.join(base_dir, actor_folder)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            parts = file.split("-")
            modality, channel, emotion, intensity, statement, repetition, actor = parts
            actor = actor.split(".")[0]

            data.append({
                "filename": file,
                "path": os.path.join(actor_path, file),
                "modality": modality,
                "channel": channel,
                "emotion_id": emotion,
                "emotion": emotion_map.get(emotion, "unknown"),
                "intensity": "strong" if intensity == "02" else "normal",
                "statement_id": statement,
                "statement": statement_map.get(statement, "unknown"),
                "repetition": repetition,
                "actor_id": actor,
                "gender": "male" if int(actor) % 2 != 0 else "female"
            })

df = pd.DataFrame(data)
df.to_csv("dataset.csv", index=False)
print(f"✅ Metadata CSV created with {len(df)} rows and {len(df.columns)} columns")
print(df.head())