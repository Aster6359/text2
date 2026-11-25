# 信息熵与图像分类实验

本项目使用模拟灰度数据和图像分类模型，探索信息熵、交叉熵与置信度，并测试双模型推理分流。

- `problem_1.py`：计算三组模拟像素数据的信息熵。
- `predict_entropy.py`：使用 MobileNet 分析图像分类预测熵，并生成概率分布图表。
- `verify_routing.py`：对比高熵与低置信度复核，并分析温度缩放前后的交叉熵和置信度。
- `benchmark_routing.py`：比较不同复核比例下的准确率、耗时和吞吐量，生成性能图表。

运行需要 Python、PyTorch、TorchVision 和 Matplotlib。首次运行 `predict_entropy.py` 会下载 Imagenette 数据集及模型权重；其余脚本使用相同的数据目录。
