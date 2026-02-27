import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class DEAMDataset(Dataset):
    def __init__(self, csv_path, mel_dir):
        self.df = pd.read_csv(csv_path)
        self.mel_dir = mel_dir

        # Collect available mel IDs
        available_ids = {
            int(f.replace(".npy", ""))
            for f in os.listdir(mel_dir)
            if f.endswith(".npy")
        }

        # Filter CSV to match available mel files
        self.df = self.df[self.df["song_id"].isin(available_ids)]
        self.df = self.df.reset_index(drop=True)

        print(f"[INFO] Dataset size after filtering: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        song_id = int(row["song_id"])
        mel_path = os.path.join(self.mel_dir, f"{song_id}.npy")

        mel = np.load(mel_path)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6 )

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

       # -------- FIXED LABEL NORMALIZATION --------
        valence = row["valence_mean"]
        arousal = row["arousal_mean"]

    # Convert 1–9 scale → [-1, 1]
        valence = (valence - 5.0) / 4.0
        arousal = (arousal - 5.0) / 4.0

        target = torch.tensor([valence, arousal], dtype=torch.float32)

        return mel, target
