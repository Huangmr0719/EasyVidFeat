#modified from 

import torch as th
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from ..utils.video_loader import VideoLoader
from torch.utils.data import DataLoader
from ..utils.preprocessing import Preprocessing
import clip
from transformers import AutoProcessor, AutoModel

class VideoFeatureExtractor:
    def __init__(
        self,
        model_type='clip',
        model_name='ViT-L/14',
        image_size=224,
        batch_size=32,
        half_precision=True,
        l2_normalize=False,
        model_path=None
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.half_precision = half_precision
        self.l2_normalize = l2_normalize
        
        # 加载模型
        if model_type == 'clip':
            self.model, _ = clip.load(model_name, download_root=model_path)
            self.processor = None
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
        self.model.eval()
        self.model = self.model.cuda()
        self.preprocess = Preprocessing()
        
    def extract_from_csv(self, csv_path):
        dataset = VideoLoader(
            csv_path,
            framerate=1,
            size=self.image_size,
            centercrop=True,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        with th.no_grad():
            for k, data in enumerate(loader):
                input_file = data["input"][0]
                output_file = data["output"][0]
                if len(data["video"].shape) > 3:
                    print(f"Computing features of video {k + 1}/{n_dataset}: {input_file}")
                    video = data["video"].squeeze()
        
                    if len(video.shape) == 4:
                        if args.model_type == 'clip':
                            video = preprocess(video)
                        else:
                            # 使用 ViT 处理器处理视频帧
                            video = th.stack([
                                th.from_numpy(
                                    processor(images=frame.numpy(), return_tensors="pt")
                                    .pixel_values.squeeze()
                                ) for frame in video
                            ])
        
                        n_chunk = len(video)
                        features = th.cuda.FloatTensor(n_chunk, args.feature_dim).fill_(0)
                        n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
        
                        for i in tqdm(range(n_iter)):
                            min_ind = i * args.batch_size
                            max_ind = (i + 1) * args.batch_size
                            video_batch = video[min_ind:max_ind].cuda()
        
                            if args.model_type == 'clip':
                                batch_features = model.encode_image(video_batch)
                            else:
                                # 提取 ViT 特征
                                outputs = model(pixel_values=video_batch)
                                batch_features = outputs.last_hidden_state[:, 0]  # 使用 [CLS] token 作为特征
        
                            if args.l2_normalize:
                                batch_features = F.normalize(batch_features, dim=1)
                            
                            features[min_ind:max_ind] = batch_features
        
                        print(f"features shape: {features.shape}")
                        features = features.cpu().numpy()
                        if args.half_precision:
                            features = features.astype("float16")
                        np.save(output_file, features)
                else:
                    print(f"Video {input_file} already processed.")
    
    def merge_features(self, feature_folder, output_path, pad_length=0):
        """合并提取的特征"""
        from ..utils.merge_features import FeatureMerger
        
        merger = FeatureMerger(pad_length=pad_length)
        return merger.merge_folder(feature_folder, output_path)