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

- Post-process keypoints to derive more meaningful features
- Make all paths constants
- Exception handling
- Document code
- Logging by module
- Inference monitoring
- Experiment tracking
- Model deployer module
- Package as library `fitml`
- Use a transformer model (Hugging Face) to post-process predictions