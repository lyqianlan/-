# -
# 人工智能科学与技术第一次大作业

## 模块 A：环形数据集神经网络实验

### 运行环境

Python 3.8 ~ 3.10

TensorFlow 2.x （本实验使用 2.10 或更高版本均可）

Scikit-learn

Matplotlib

NumPy

### 依赖安装

pip install tensorflow scikit-learn matplotlib numpy

### 数据集

脚本自动生成环形数据集（make_circles），无需手动下载。

### 执行顺序

### 直接运行脚本即可：

生成 1000 个环形样本（无噪声），按 1:1 划分训练/测试。

实验一：对比 6 种网络结构（1层2/4神经元，2层4/8神经元，3层4/8神经元），训练 100 个 epoch，记录训练/测试损失。

实验二：固定 3 层 4 神经元结构，对比 ReLU、Tanh、Sigmoid、Linear 四种激活函数，训练 500 个 epoch。

输出结果表格并绘制最佳模型的决策边界图。

### 预期输出

控制台打印两种实验的损失表格。

弹出决策边界可视化图像（保存后可关闭）。

## 模块 B：ResNet50+FPN 车辆多目标检测

### 运行环境
Python 3.8 ~ 3.10

PyTorch 2.x（建议 2.0 以上）

torchvision

OpenCV-Python

pycocotools

COCO 2017 数据集


### 依赖安装

pip install torch torchvision opencv-python pycocotools

若需使用自动混合精度（AMP），PyTorch 2.0 已内置，无需额外安装。

### 数据集准备
从 COCO 官网 下载 2017 版数据：

train2017.zip（~18GB）

val2017.zip（~1GB）

annotations_trainval2017.zip（~241MB）

解压到 ./coco2017/ 目录下，确保目录结构为：

coco2017/

├── train2017/

├── val2017/

└── annotations/

    ├── instances_train2017.json
    
    └── instances_val2017.json
    
如果数据集路径不同，请修改 data_root 变量。

### 执行顺序

train

vision

### 训练流程：

加载 COCO 车辆子集（5 类车辆：car, motorcycle, bus, train, truck）。

构建 ResNet50+FPN 的 Faster R-CNN 模型，并使用 ImageNet 预训练权重初始化。

训练 12 个 epoch，批量大小 4（训练）/2（验证），学习率初始 0.02，在第 8 和 11 epoch 衰减 0.1 倍。

每 2 个 epoch 在验证集上计算 COCO 标准 mAP、AP50、AP_S 等指标，并保存最佳模型。

训练结束后保存最终模型 final_vehicle_detector.pth。



