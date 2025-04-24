import argparse
from .extractors.visual_extract import VideoFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="视频特征提取工具")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 特征提取命令
    extract_parser = subparsers.add_parser('extract', help='提取视频特征')
    extract_parser.add_argument("--csv", type=str, required=True,
                              help="输入CSV文件路径")
    extract_parser.add_argument("--model_type", type=str, default='clip',
                              choices=['clip', 'huggingface'],
                              help="选择模型类型")
    extract_parser.add_argument("--model_name", type=str, default='ViT-L/14',
                              help="模型名称")
    extract_parser.add_argument("--image_size", type=int, default=224,
                              help="输入图像尺寸")
    extract_parser.add_argument("--batch_size", type=int, default=32,
                              help="批处理大小")
    
    # 特征合并命令
    merge_parser = subparsers.add_parser('merge', help='合并提取的特征')
    merge_parser.add_argument("--folder", type=str, required=True,
                            help="特征文件夹路径")
    merge_parser.add_argument("--output", type=str, required=True,
                            help="输出文件路径")
    merge_parser.add_argument("--pad", type=int, default=0,
                            help="填充/截断长度")
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extractor = VideoFeatureExtractor(
            model_type=args.model_type,
            model_name=args.model_name,
            image_size=args.image_size,
            batch_size=args.batch_size
        )
        extractor.extract_from_csv(args.csv)
        
    elif args.command == 'merge':
        extractor = VideoFeatureExtractor()
        extractor.merge_features(
            args.folder,
            args.output,
            pad_length=args.pad
        )

if __name__ == "__main__":
    main()