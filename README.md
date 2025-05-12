# doubl-ml

## Assumptions

- The images keypoint extraction is already done, this is part of the data preparation phase of training
- The data distribution is know to be able to generate synthetic data

## Data

- Images downloaded from Unsplash

## How to run

- cd to `doubl-ml`
- `pip install -e .`
- `python .\fitml\train.py`

## Next steps

- Use a transformer model (Hugging Face and later bigger models) to post-process predictions and deliver natural
  language recommendations
- Post-process keypoints to derive more meaningful features to improve generalization predictions
- Turn feature extraction into neural network layers to offload processing from hardware to model
- Exception handling, code documentation, logging by module
- Model inference monitoring
- Experiment tracking
- Model deployer module
