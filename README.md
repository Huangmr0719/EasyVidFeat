# Easy Video Feature Extraction

A simple and user-friendly tool for video feature extraction, supporting various pretrained models including CLIP and Hugging Face models.

## Features
- Support for multiple pretrained models (CLIP, Hugging Face)
- Batch processing of video files
-	Customizable image size and batch size
-	Optional feature normalization
-	Support for half-precision (FP16) output
-	Automatic management of temporary files

## Installation

### Install from source

```bash
git clone https://github.com/Huangmr0719/EasyVidFeat.git
cd EasyVidFeat
pip install -e .
```

## Usage

### 1. Command-line Usage

Extract features:

```bash
video-extract extract \
    --video_root your_video_path \
    --feature_root your_feature_path \
    --feature_name clip_features \
    --model_type clip \
    --model_name ViT-L/14
```

Merge features:

```bash
video-extract merge \
    --feature_root your_feature_path \
    --feature_name clip_features \
    --pad 100
```

Full argument example:

```bash
video-extract --video_root your_video_path \

              --model_type clip \
              --model_name ViT-L/14 \
              --image_size 224 \
              --batch_size 32 \
              --half_precision 1 \
              --l2_normalize 0
```

### Parameters
#### Extract Command
-	`--video_root` : Root directory of videos
-  `--feature_root` : Root directory to save features
-  `--feature_name` : Feature Name
-	`--model_type` : Model type, choose from 'clip' or 'huggingface'
-	`--model_name` : Name of the model to use
-	`--image_size` : Input image size (default: 224)
-	`--batch_size` : Batch size for processing (default: 32)
-  `--model_path` : Path to the model file (for offline models)
-	`--half_precision` : Use half-precision (FP16) if set to 1 (default: 1)
-	`--l2_normalize` : Apply L2 normalization to features if set to 1 (default: 0)
-  `--framerate` : Video sampling framerate (default: 1)
-  `--num_workers` : Number of data loading workers (default: 4)
-  `--device` : Device to run on (default: cuda)

#### Merge Command
- `--feature_root`: Root directory of features
- `--feature_name`: Name of feature directory
- `--pad`: Padding/truncation length (default: 0)

### 2. Python API Usage

#### Basic Usage

```python
from easy_video_extract import VideoFeatureExtractor

# Initialize the extractor
extractor = VideoFeatureExtractor(
    model_type='clip',
    model_name='ViT-L/14'
)

# Extract features
extractor.extract_from_csv('videos.csv')

# Merge features
extractor.merge_features(
    feature_folder='features_folder',
    output_path='merged_features.pt',
    pad_length=100  # Optional: set padding/truncation length
)
```

#### Advanced Usage

```python
extractor = VideoFeatureExtractor(
    model_type='huggingface',
    model_name='BAAI/EVA-CLIP',
    image_size=336,
    batch_size=16,
    half_precision=True,
    l2_normalize=True,
    model_path='path/to/model'  # Optional: for offline models
)
```


### 3. CSV File Format

The CSV file should contain the following columns:

```bash
video_path,feature_path
/path/to/video1.mp4,/path/to/features/video1.npy
/path/to/video2.mp4,/path/to/features/video2.npy
```

## Supported Models

**CLIP Models**

-	ViT-B/32
-	ViT-B/16
-	ViT-L/14
-	ViT-L/14@336px

**Hugging Face Models**

- BAAI/EVA-CLIP
- openai/clip-vit-base-patch32
- google/vit-base-patch16-224
- Other compatible vision models


## Notes
### 1.	Memory Usage

- Batch size directly impacts memory consumption
- For high-resolution videos, consider reducing the batch size
  
### 2.	GPU Requirements

-	CUDA-compatible NVIDIA GPU recommended
-	Minimum 4GB VRAM (8GB or more recommended)
  
### 3.	Feature Dimensions

-	CLIP ViT-L/14: 768-dimensional
-	EVA-CLIP: 1024-dimensional
-	Other models may vary


## FAQ
### 1. Out of memory error
 
```python
# Reduce batch size
extractor = VideoFeatureExtractor(batch_size=8)
```

### 2.	Using CPU

```python
# Move model to CPU (not recommended due to slower performance)
extractor.model = extractor.model.cpu()
```

### 3.	Custom preprocessing

```python
# Change input image size
extractor = VideoFeatureExtractor(image_size=336)
```



## License

MIT License

## Citation

If you use this tool in your research, please cite it as:

```
@software{easy_video_extract,
  title = {Easy Video Feature Extraction},
  author = {Huangmr0719},
  year = {2025},
  url = {https://github.com/Huangmr0719/EasyVidFeat}
}
```
