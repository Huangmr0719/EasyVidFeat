import os
from pathlib import Path
import csv

def make_path_csv(video_root_dir, feature_root_dir, feature_name, output_path=None):
    """
    为视频目录下的所有视频文件生成路径映射CSV文件

    Args:
        video_root_dir (str): 视频根目录路径
        feature_root_dir (str): 特征保存根目录路径
        feature_name (str): 特征目录名称
        output_path (str, optional): CSV文件保存路径，若为None则保存在.tmp目录下
    """
    # 支持的视频格式
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv')
    
    # 确保路径是Path对象
    video_root = Path(video_root_dir)
    feature_root = Path(feature_root_dir) / feature_name
    
    # 设置默认输出路径
    if output_path is None:
        tmp_dir = Path(os.getcwd()) / '.tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        output_path = tmp_dir / f"{feature_name}_paths.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化CSV数据
    data = [["video_path", "feature_path"]]
    video_count = 0
    
    # 遍历视频目录
    for video_path in video_root.rglob("*"):
        if video_path.suffix.lower() in VIDEO_EXTENSIONS:
            rel_path = video_path.relative_to(video_root)
            feature_path = feature_root / rel_path.parent / video_path.stem
            data.append([str(video_path), str(feature_path)])
            video_count += 1
            
            if video_count % 100 == 0:
                print(f"已处理 {video_count} 个视频文件...")
    
    # 保存CSV文件
    try:
        with open(output_path, "w", newline="", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerows(data)
        print(f"已成功处理 {video_count} 个视频文件")
        print(f"CSV文件已保存至：{output_path}")
    except Exception as e:
        print(f"保存CSV文件时出错：{e}")

if __name__ == '__main__':
    # 使用示例
    video_root_dir = r"D:\Videos"  # 视频根目录
    feature_root_dir = r"D:\Features"  # 特征保存根目录
    feature_name = "clip_features"  # 特征目录名称
    
    # 使用默认路径（保存在.tmp目录下）
    make_path_csv(video_root_dir, feature_root_dir, feature_name)
    
    # 或指定保存路径
    # make_path_csv(video_root_dir, feature_root_dir, feature_name, r"D:\my_paths.csv")