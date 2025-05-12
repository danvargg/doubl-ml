import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

LABEL_MAP_REV = {0: "tight", 1: "ideal", 2: "relaxed"}


class FitPredictor:
    def __init__(self, model_path: str):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def _extract_keypoints(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if not results.pose_landmarks:
            raise ValueError("No pose detected.")
        keypoints = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
        return keypoints

    def predict(self, image_path: str, user_meta: dict, garment_meta: dict) -> tuple:
        print("making prediction...")
        keypoints = self._extract_keypoints(image_path)

        pref_map = {"slim": 0, "regular": 1, "loose": 2}
        cut_map = {"tight": 0, "regular": 1, "relaxed": 2}

        user_vec = [
            user_meta["height_cm"],
            user_meta["weight_kg"],
            pref_map[user_meta["fit_preference"]]
        ]

        garment_vec = [
            garment_meta["shoulder_width_cm"],
            garment_meta["waist_cm"],
            garment_meta["length_cm"],
            cut_map[garment_meta["cut"]]
        ]

        x = np.concatenate([keypoints, user_vec, garment_vec]).astype(np.float32).reshape(1, -1)
        pred = self.model.predict(x)
        label_idx = np.argmax(pred, axis=1)[0]
        label = LABEL_MAP_REV[label_idx]

        explanation = self._generate_explanation(user_meta, garment_meta)
        return label, explanation

    def _generate_explanation(self, user_meta, garment_meta) -> str:  # FIXME: use language model for recommendation
        print("generating explanation...")
        diff = user_meta["waist_cm"] - garment_meta["waist_cm"]
        if abs(diff) < 2:
            return f"Waist measurement is well-aligned with garment size ({diff:+.1f}cm difference)."
        elif diff > 0:
            return f"Waist measurement exceeds garment size by {diff:.1f}cm."
        else:
            return f"Garment waist is larger than user's waist by {-diff:.1f}cm."


def main():
    image_path = "data/images/shot05.jpg"

    user_meta = {
        "height_cm": 175,
        "weight_kg": 70,
        "fit_preference": "regular",
        "waist_cm": 82
    }

    garment_meta = {
        "shoulder_width_cm": 44,
        "waist_cm": 78,
        "length_cm": 70,
        "cut": "regular"
    }

    model_path = "trained_model/predictor_tf.h5"

    predictor = FitPredictor(model_path)

    fit, explanation = predictor.predict(image_path, user_meta, garment_meta)

    print(f"Predicted fit: {fit}")
    print(f"Explanation: {explanation}")


if __name__ == "__main__":
    main()
