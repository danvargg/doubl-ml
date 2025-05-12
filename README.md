# fitml

## Project Overview

This system predicts how well a garment fits a user, based on:

- Inputs: A full-body front-facing image (pose keypoints extracted)
- User metadata: Height (cm), weight (kg), fit preference (slim, regular, loose)
- Garment metadata: Shoulder width, waist, length (cm), cut type (tight, regular, relaxed)
- Outputs: A predicted fit category: "tight", "ideal", or "relaxed"
- A justification string (e.g., "waist measurement exceeds garment size by 4cm")

The code has been packaged into a python library for ease of use.

## Assumptions

- The images are properly taken for keypoint extraction
- The data distribution is know to be able to generate synthetic data
- The dataset is properly labeled

## Data

- Images: Royalty-free full-body photos from Unsplash, processed with `MediaPipe`
- User metadata: Manually simulated (see `metadata.py`)
- Garment metadata: Manually simulated (see `metadata.py`)

## Code structure

- `fitml/data.py`: Data loader and preprocessing logic
- `fitml/inference.py`: Inference script (loads model and outputs prediction + explanation)
- `fitml/metadata.py`: Dummy data for training
- `fitml/model.py`: Model building and management
- `fitml/train.py`: Model training script

## Sample output

```bash
Predicted fit: tight
Explanation: Waist measurement exceeds garment size by 4.0cm.
```

## How to run

### Train the model

- cd to `doubl-ml`
- `pip install -e .`
- `python .\fitml\train.py`

### Predict on new data

- Update `image_path = "data/images/shot05.jpg"` in `fitml\inference.py`
- `python .\fitml\inference.py`

## Next steps

- Generate natural language recommendations using a language model (e.g., Hugging Face) to translate predictions into
  user-friendly advice
- Develop `fastAPI` module to serve a website load
- Engineer more meaningful features from pose keypoints to improve model generalization (e.g., inferred body ratios,
  distances)
- Integrate feature extraction into the model architecture (e.g., using custom layers) to shift preprocessing from
  code to model
- Add exception handling, logging, and inline documentation across modules for robustness and maintainability
- Implement model inference monitoring to track performance in real-world usage
- Integrate experiment tracking using tools like `MLflow` or `Weights & Biases`
- Build a model deployment module for serving the model via embedded environments
- Simulate variation in pose data (e.g., noise, slight scale/rotation changes) to make the model more robust to
  real-world inputs
- Let users rate fit recommendations, and use this feedback for continual learning or re-training
- Plot pose keypoints or inferred body/garment measurements on images to help debug and explain predictions visually