from setuptools import setup, find_packages

setup(
    name="EasyVidFeat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy",
        "opencv-python",
        "tqdm",
        "clip",
        "transformers",
        "Pillow",
        "pandas"
    ],
    entry_points={
        'console_scripts': [
            'video-extract=easy_video_extract.cli:main',
        ],
    },
    author="Mingru Huang",
    author_email="huangmr0719@outlook.com",
    description="一个简单的视频特征提取工具",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/EasyVidFeat",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)