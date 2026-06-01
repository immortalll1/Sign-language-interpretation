import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class SignLanguageDataset(Dataset):
    def __init__(
        self,
        csv_path,
        video_dir,
        sequence_length=16,
        img_size=112,
        train=True,
        verbose=True,
        use_cache=False,
        augment=False
    ):
        self.video_dir = os.path.abspath(video_dir)
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.train = train
        self.verbose = verbose
        self.use_cache = use_cache
        self.augment = augment and train
        self.cache = {}

        csv_path = os.path.abspath(csv_path)
        full_data = self._read_and_prepare_csv(csv_path)

        self.classes = sorted(full_data["text"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.data = full_data[full_data["train"] == bool(train)].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(
                f"После фильтрации train={train} не осталось данных. "
                f"Проверь значения в столбце train."
            )

        if verbose:
            split_name = "train" if train else "test"
            print(f"[{split_name}] Размер датасета: {len(self.data)}")
            print(f"[{split_name}] Количество классов: {len(self.classes)}")
            print(f"[{split_name}] Первые 20 классов: {self.classes[:20]}")

    def _read_and_prepare_csv(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV не найден: {csv_path}")

        df = pd.read_csv(
            csv_path,
            sep=";",
            engine="python",
            encoding="cp1251"
        )

        df.columns = (
            df.columns.astype(str)
            .str.replace('"', '', regex=False)
            .str.replace("'", "", regex=False)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
        )

        rename_map = {}
        for col in df.columns:
            clean_col = col.replace('"', '').replace("'", "").strip().lower()
            if clean_col.startswith("attachment_id"):
                rename_map[col] = "attachment_id"
            elif clean_col.startswith("text"):
                rename_map[col] = "text"
            elif clean_col.startswith("train"):
                rename_map[col] = "train"
            elif clean_col.startswith("begin"):
                rename_map[col] = "begin"
            elif clean_col.startswith("end"):
                rename_map[col] = "end"

        df = df.rename(columns=rename_map)

        required = ["attachment_id", "text", "train", "begin", "end"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"В CSV отсутствует столбец {col}")

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace('"', '', regex=False)
                    .str.replace("'", "", regex=False)
                    .str.strip()
                )

        df["train"] = (
            df["train"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "истина": True,
                "ложь": False
            })
        )

        df["begin"] = pd.to_numeric(df["begin"], errors="coerce").fillna(0).astype(int)
        df["end"] = pd.to_numeric(df["end"], errors="coerce").fillna(9999).astype(int)

        df = df.dropna(subset=["attachment_id", "text", "train"]).copy()
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def apply_augmentation(self, frame_rgb: np.ndarray) -> np.ndarray:
        if not self.augment:
            return frame_rgb

        frame = frame_rgb.copy()

        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.9, 1.1)
            frame = np.clip(frame * alpha, 0, 255).astype(np.uint8)

        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame

    def load_video(self, path, begin, end):
        cache_key = (path, begin, end, self.sequence_length, self.img_size, self.augment)

        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Видео не найдено: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {path}")

        begin = max(0, int(begin))
        end = max(begin, int(end))
        cap.set(cv2.CAP_PROP_POS_FRAMES, begin)

        frames = []
        frame_id = begin

        while frame_id <= end:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.apply_augmentation(frame)
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)
            frame_id += 1

        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)]

        if len(frames) >= self.sequence_length:
            ids = np.linspace(0, len(frames) - 1, self.sequence_length).astype(int)
            frames = [frames[i] for i in ids]
        else:
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])

        frames = np.array(frames, dtype=np.float32)
        frames = np.transpose(frames, (3, 0, 1, 2))

        video_tensor = torch.tensor(frames, dtype=torch.float32)

        if self.use_cache:
            self.cache[cache_key] = video_tensor

        return video_tensor

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        video_name = str(row["attachment_id"]).strip() + ".mp4"
        video_path = os.path.join(self.video_dir, video_name)

        begin = int(row["begin"])
        end = int(row["end"])

        video = self.load_video(video_path, begin, end)
        label = self.class_to_idx[row["text"]]

        return video, label