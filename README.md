# Easy Video Feature Extraction

一个简单易用的视频特征提取工具，支持多种预训练模型，包括 CLIP 和 Hugging Face 模型。

## 功能特点

- 支持多种预训练模型（CLIP、Hugging Face）
- 批量处理视频文件
- 自定义图像尺寸和批处理大小
- 支持特征归一化
- 支持半精度（FP16）输出
- 自动管理临时文件

## 安装

### 从源码安装

```bash
git clone https://github.com/yourusername/EasyVidFeat.git
cd EasyVidFeat
pip install -e .
```

## 使用方法

### 1. 命令行使用

提取特征：
```bash
video-extract extract --csv videos.csv --model_type clip --model_name ViT-L/14
```

合并特征：
```bash
video-extract merge --folder features_folder --output merged_features.pt --pad 100
```

完整参数说明：
```bash
video-extract --csv videos.csv \
              --model_type clip \
              --model_name ViT-L/14 \
              --image_size 224 \
              --batch_size 32 \
              --half_precision 1 \
              --l2_normalize 0
```

参数说明：
- `--csv`: 包含视频路径和输出路径的CSV文件
- `--model_type`: 模型类型，可选 'clip' 或 'huggingface'
- `--model_name`: 模型名称
- `--image_size`: 输入图像尺寸（默认224）
- `--batch_size`: 批处理大小（默认32）
- `--half_precision`: 是否使用半精度（默认1）
- `--l2_normalize`: 是否进行L2归一化（默认0）

### 2. Python API 使用

基本用法：
```python
from easy_video_extract import VideoFeatureExtractor

# 初始化提取器
extractor = VideoFeatureExtractor(
    model_type='clip',
    model_name='ViT-L/14'
)

# 提取特征
extractor.extract_from_csv('videos.csv')

# 合并特征
extractor.merge_features(
    feature_folder='features_folder',
    output_path='merged_features.pt',
    pad_length=100  # 可选，设置填充/截断长度
)
```

高级用法：
```python
extractor = VideoFeatureExtractor(
    model_type='huggingface',
    model_name='BAAI/EVA-CLIP',
    image_size=336,
    batch_size=16,
    half_precision=True,
    l2_normalize=True,
    model_path='path/to/model'  # 可选，用于离线模型
)
```

### 3. CSV文件格式

CSV文件需要包含以下列：
```csv
video_path,feature_path
/path/to/video1.mp4,/path/to/features/video1.npy
/path/to/video2.mp4,/path/to/features/video2.npy
```

## 支持的模型

### CLIP 模型
- ViT-B/32
- ViT-B/16
- ViT-L/14
- ViT-L/14@336px

### Hugging Face 模型
- BAAI/EVA-CLIP
- openai/clip-vit-base-patch32
- google/vit-base-patch16-224
- 其他兼容的视觉模型

## 注意事项

1. 内存使用：
   - 批处理大小会影响内存使用
   - 对于高分辨率视频，建议适当减小批处理大小

2. GPU 要求：
   - 推荐使用 CUDA 支持的 NVIDIA GPU
   - 至少 4GB 显存（推荐 8GB 以上）

3. 特征维度：
   - CLIP ViT-L/14: 768维
   - EVA-CLIP: 1024维
   - 其他模型维度可能不同

## 常见问题

1. 内存不足：
   ```python
   # 减小批处理大小
   extractor = VideoFeatureExtractor(batch_size=8)
   ```

2. 使用CPU：
   ```python
   # 将模型移至CPU（不推荐，会很慢）
   extractor.model = extractor.model.cpu()
   ```

3. 自定义预处理：
   ```python
   # 修改图像尺寸
   extractor = VideoFeatureExtractor(image_size=336)
   ```

## 许可证

MIT License

## 引用

如果您在研究中使用了本工具，请引用：

```bibtex
@software{easy_video_extract,
  title = {Easy Video Feature Extraction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/EasyVidFeat}
}
```
