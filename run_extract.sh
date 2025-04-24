# extract features
CUDA_VISIBLE_DEVICES=0 python -m easy_video_extract extract \
--video_root D:\videos \
--feature_root D:\features \
--feature_name clip_features \
--model_path D:\models \
--model_type clip \
--model_name ViT-L/14

# merge features
python -m easy_video_extract merge \
--feature_root D:\features \
--feature_name clip_features \
--pad 100
