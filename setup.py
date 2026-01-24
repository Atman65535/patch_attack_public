from setuptools import setup, find_packages

setup(
    name="patch_attack",             # 包的名字，随便起
    version="0.1.0",               # 版本号
    packages=find_packages(),      # 核心：自动发现包含 __init__.py 的文件夹（如 usr）
    install_requires=[             # 可选：列出必要的依赖，执行 pip install . 时会自动装
        "torch",
        "segmentation-models-pytorch",
        "diffusers",
        "transformers",
    ],
    python_requires=">=3.8",
)