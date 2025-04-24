import argparse
import os
from .extractors.visual_extract import VideoFeatureExtractor
from .utils.make_path_csv import make_path_csv

def main():
    parser = argparse.ArgumentParser(description="Video Feature Extraction Tool")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract video features')
    extract_parser.add_argument("--video_root", type=str, required=True,
                              help="Root directory of videos")
    extract_parser.add_argument("--feature_root", type=str, required=True,
                              help="Root directory to save features")
    extract_parser.add_argument("--feature_name", type=str, required=True,
                              help="Name of feature directory")
    extract_parser.add_argument("--model_type", type=str, default='clip',
                              choices=['clip', 'huggingface'],
                              help="Type of model to use")
    extract_parser.add_argument("--model_name", type=str, default='ViT-L/14',
                              help="Name of the model")
    extract_parser.add_argument("--image_size", type=int, default=224,
                              help="Input image size")
    extract_parser.add_argument("--batch_size", type=int, default=32,
                              help="Batch size for processing")
    extract_parser.add_argument("--model_path", type=str, default=None,
                              help="Path to model weights")
    extract_parser.add_argument("--half_precision", type=bool, default=True,
                              help="Use half precision")
    extract_parser.add_argument("--l2_normalize", type=bool, default=False,
                              help="Apply L2 normalization")
    extract_parser.add_argument("--framerate", type=int, default=1,
                              help="Video sampling framerate")
    extract_parser.add_argument("--num_workers", type=int, default=4,
                              help="Number of data loading workers")
    extract_parser.add_argument("--device", type=str, default="cuda",
                              help="Device to run on")
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge extracted features')
    merge_parser.add_argument("--feature_root", type=str, required=True,
                            help="Root directory of features")
    merge_parser.add_argument("--feature_name", type=str, required=True,
                            help="Name of feature directory")
    merge_parser.add_argument("--pad", type=int, default=0,
                            help="Padding/truncation length")
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        # Generate CSV file
        csv_path = os.path.join(args.feature_root, f"{args.feature_name}_paths.csv")
        make_path_csv(args.video_root, args.feature_root, args.feature_name, csv_path)
        
        # Extract features
        extractor = VideoFeatureExtractor(
            model_type=args.model_type,
            model_name=args.model_name,
            image_size=args.image_size,
            batch_size=args.batch_size,
            half_precision=args.half_precision,
            l2_normalize=args.l2_normalize,
            model_path=args.model_path,
            device=args.device
        )
        extractor.extract_from_csv(csv_path, framerate=args.framerate, num_workers=args.num_workers)
        
    elif args.command == 'merge':
        extractor = VideoFeatureExtractor()
        feature_folder = os.path.join(args.feature_root, args.feature_name)
        output_path = os.path.join(args.feature_root, f"{args.feature_name}.pth")
        extractor.merge_features(feature_folder, output_path, pad_length=args.pad)

if __name__ == "__main__":
    main()