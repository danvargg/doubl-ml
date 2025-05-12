import glob
import os

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from metadata import users, garment


class DataLoader:
    def __init__(
            self,
            images_dir: str = "data/images",
            user_meta_path: str = "data/users.json",
            garment_meta_path: str = "data/garment.json",
            batch_size: int = 2,
            shuffle: bool = True
    ):
        self.images_dir = images_dir
        self.user_meta_path = user_meta_path
        self.garment_meta_path = garment_meta_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))

        self.users = users
        self.garments = garment

        self.pref_map = {"slim": 0, "regular": 1, "loose": 2}
        self.cut_map = {"tight": 0, "regular": 1, "relaxed": 2}
        self.label_map = {"tight": 0, "ideal": 1, "relaxed": 2}

        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def _extract_keypoints(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if not results.pose_landmarks:
            raise ValueError(f"No landmarks found for {image_path}")

        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32).flatten()
        return keypoints

    def _make_samples(self):
        for i, image_path in enumerate(self.image_paths):
            try:
                keypoints = self._extract_keypoints(image_path)
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
                continue

            user = self.users[i % len(self.users)]
            fit_pref = user.get("fit_preference", "regular")
            if fit_pref not in self.pref_map:
                print(f"Skipping user with unknown fit_preference: {fit_pref}")
                continue

            user_vec = [
                user["height_cm"],
                user["weight_kg"],
                self.pref_map[fit_pref]
            ]

            for garment in self.garments:
                cut = garment.get("cut", "regular")
                if cut not in self.cut_map:
                    print(f"Skipping garment with unknown cut: {cut}")
                    continue

                garment_vec = [
                    garment["shoulder_width_cm"],
                    garment["waist_cm"],
                    garment["length_cm"],
                    self.cut_map[cut]
                ]

                label = np.random.randint(0, 3)
                x = np.concatenate([keypoints, user_vec, garment_vec]).astype(np.float32)
                y = np.array(label, dtype=np.int32)
                yield x, y

    def get_dataset(self):
        print("get dataset")
        x_sample, y_sample = next(self._make_samples())

        ds = tf.data.Dataset.from_generator(
            self._make_samples,
            output_signature=(
                tf.TensorSpec(shape=x_sample.shape, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )

        if self.shuffle:
            ds = ds.shuffle(buffer_size=100)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
