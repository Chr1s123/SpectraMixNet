# SpectraMixNet

基于 PyTorch 的图像去雾深度学习模型 (SpectraMixNet)。

## 环境要求

请确保已安装 Python 环境，并安装所需的依赖项：

```bash
pip install -r requirements.txt
```

**主要依赖:**
- torch
- torchvision
- opencv_python
- timm
- ptflops
- pytorch_msssim

## 数据集结构

默认情况下，代码假设数据集位于 `../Dehazing/data/` 目录下。您可以通过命令行参数修改数据路径。

推荐的数据集目录结构 (以 RESIDE-IN 为例):

```
../Dehazing/data/RESIDE-IN/
├── train/
│   ├── source/      # 有雾图像
│   └── target/      # 清晰图像
└── test/
    ├── source/
    └── target/
```

## 使用说明

### 1. 训练 (Training)

使用 `train1.py` 脚本进行模型训练。

```bash
# 示例：使用 SpectraMixNet-t 模型在 indoor 数据集上训练
python train1.py --model SpectraMixNet-t --exp indoor --gpu 0,1
```

**常用参数:**
- `--model`: 模型名称 (默认: `SpectraMixNet-t`)
- `--exp`: 实验名称/数据集类型 (默认: `indoor`)
- `--data_dir`: 数据集根目录 (默认: `../Dehazing/data/RESIDE-IN/`)
- `--gpu`: 指定使用的 GPU ID (默认: `0,1`)

### 2. 测试 (Testing)

使用 `testNet.py` 脚本进行模型测试和评估。

```bash
# 示例：测试训练好的模型
python testNet.py --model SpectraMixNet-t --exp indoor --dataset RESIDE-IN --gpu 0
```

**注意:** 请确保 `--model` 和 `--exp` 参数与训练时保持一致，以便正确加载模型权重 (`saved_models/indoor/SpectraMixNet-t_best.pth`)。

## 结果

- 训练日志将保存在 `../Dehazing/logs/`。
- 模型权重将保存在 `./saved_models/`。
- 测试结果（图片和指标）将保存在 `./results/`。
