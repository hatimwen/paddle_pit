# 基于Paddle实现PiT  ——Rethinking Spatial Dimensions of Vision Transformers,[arxiv](https://arxiv.org/pdf/2103.16302v2.pdf)

- 官方原版代码（基于PyTorch）[pit](https://github.com/naver-ai/pit).

- 本项目基于 [PaddleViT](https://github.com/BR-IDL/PaddleViT.git) 实现，在其基础上与原版代码实现了更进一步的对齐，并通过完整训练与测试完成对pit_ti模型的复现.

## 1. 简介

从CNN的成功设计原理出发，作者研究了空间尺寸转换的作用及其在基于Transformer的体系结构上的有效性。

具体来说，类似于CNN的降维原则（随着深度的增加，传统的CNN会增加通道尺寸并减小空间尺寸），作者用实验表明了这同样有利于Transformer的性能提升，并提出了基于池化的Vision Transformer，即PiT（模型示意图如下）。

<p align="center">
<img src="./images/pit.png" alt="drawing" width="90%" height="90%"/>
    <h4 align="center">PiT 模型示意图</h4>
</p>

## 2. 数据集和复现精度

### 数据集

原文使用的为ImageNet-1k 2012（ILSVRC2012），共1000类，训练集/测试集图片分布：1281167/50000，数据集大小为144GB。

本项目使用的为官方推荐的图片压缩过的更轻量的Light_ILSVRC2012，数据集大小为65GB。其在AI Studio上的地址为：[Light_ILSVRC2012_part_0.tar](https://aistudio.baidu.com/aistudio/datasetdetail/114241) 与 [Light_ILSVRC2012_part_1.tar](https://aistudio.baidu.com/aistudio/datasetdetail/114746)。

### 复现精度

|  Model   |  目标精度Acc@1|  实现精度Acc@1|Image Size   | batch_size | Crop_pct   | epoch  |#Params|
|  ----    |  ----      |  ----     |  ----       | ----       | ----       | ----   | ----  |
| pit_ti  |  73.0       |  **73.01**     |224          |256*4GPUs         |0.9         | 300 <br> (+10 COOLDOWN) |4.8M|

> 【注】上表中的实现精度在原版ILSVRC2012验证集上测试得到。
值得一提的是，本项目在Light_ILSVRC2012的验证集上的Validation Acc@1达到了**73.17**。

本项目训练得到的最佳模型参数与训练日志log均存放于[output](output)文件夹下。

### 日志文件说明

本项目通过AI Studio的脚本任务运行，中途中断了4次，因此共有5个日志文件。为了方便检阅，本人手动将log命名为`log_开始epoch-结束epoch.txt`格式。具体来说：

- `output/log_1-76.txt`：epoch1~epoch76。这一版代码定义每10个epoch保存一次模型权重，每2个epoch验证一次，同时若验证精度高于历史精度，则保存为`Best_PiT.pdparams`，因此在epoch76训练结束但还未验证的时候中断，下一次的训练只能从验证精度最高的epoch74继续训练。

- `output/log_75-142.txt`：epoch75~epoch142。从这一版代码开始，新增了每次训练之后都保存一下模型参数为`PiT-Latest.pdparams`，这样无论哪个epoch训练中断都可以继续训练啦。

- `output/log_143-225.txt`：epoch143~epoch225。

- `output/log_226-303.txt`：epoch226~epoch303。

- `output/log_304-310.txt`：epoch304~epoch310。

- `output/log_eval.txt`：使用训练得到的最好模型（epoch308）在原版ILSVRC2012验证集上验证日志。

## 3. 准备环境

推荐环境配置：

- Python>=3.6
- yaml>=0.2.5
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)>=2.1.0
- [yacs](https://github.com/rbgirshick/yacs)==0.1.8
- scipy
- pyyaml

本人环境配置：

- 硬件：Tesla V100 * 4（由衷感谢百度飞桨平台提供高性能算力支持）

- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)==2.2.1
- Python==3.7

## 4. 快速开始

本项目现已通过脚本任务形式部署到AI Studio上，您可以选择fork下来直接运行`sh run.sh`，数据集处理等脚本均已部署好。链接：[paddle_pit](https://aistudio.baidu.com/aistudio/clusterprojectdetail/3397849)。

或者您也可以git本repo在本地运行：

### 第一步：克隆本项目

```
git clone https://github.com/hatimwen/paddle_pit.git
cd paddle_pit
```

### 第二步：修改参数

请根据实际情况，修改`scripts`路径下的脚本内容（如：gpu，数据集路径data_path，batch_size等）。

### 第三步：验证模型

多卡请运行：
```
sh scripts/run_eval_multi.sh
```

单卡请运行：
```
sh scripts/run_eval.sh
```

### 第四步：训练模型

多卡请运行：
```
sh scripts/run_train_multi.sh
```

单卡请运行：
```
sh scripts/run_train.sh
```

## 5.代码结构

```
|-- paddle_pit
    |-- output              # 日志及模型文件
    |-- configs             # 参数
        |-- pit_ti.yaml
    |-- datasets
        |-- ImageNet1K      # 数据集路径
    |-- scripts             # 运行脚本
        |-- run_train.sh
        |-- run_train_multi.sh
        |-- run_eval.sh
        |-- run_eval_multi.sh
    |-- augment.py          # 数据增强
    |-- config.py           # 最底层配置文件
    |-- datasets.py         # dataset与dataloader
    |-- droppath.py         # droppath定义
    |-- losses.py           # loss定义
    |-- main_multi_gpu.py   # 多卡训练测试代码
    |-- main_single_gpu.py  # 单卡训练测试代码
    |-- mixup.py            # mixup定义
    |-- model_ema.py        # EMA定义
    |-- pit.py              # pit模型结构定义
    |-- random_erasing.py   # random_erasing定义
    |-- regnet.py           # 教师模型定义（本项目并未对此验证，仅作保留）
    |-- transforms.py       # RandomHorizontalFlip定义
    |-- utils.py            # CosineLRScheduler及AverageMeter定义
    |-- README.md
    |-- requirements.txt
```

## 6. 参考及引用

```
@InProceedings{Yuan_2021_ICCV,
    author    = {Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Jiang, Zi-Hang and Tay, Francis E.H. and Feng, Jiashi and Yan, Shuicheng},
    title     = {Tokens-to-Token ViT: Training Vision Transformers From Scratch on ImageNet},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {558-567}
}
```

- [pit](https://github.com/naver-ai/pit)

- [Paddle](https://github.com/paddlepaddle/paddle)

- [PaddleViT](https://github.com/BR-IDL/PaddleViT.git)

最后，非常感谢百度举办的[飞桨论文复现挑战赛（第五期）](https://aistudio.baidu.com/aistudio/competition/detail/126/0/introduction)让本人对Paddle理解更加深刻。
同时也非常感谢[朱欤老师](https://github.com/xperzy)团队用[Paddle](https://github.com/paddlepaddle/paddle)实现的[PaddleViT](https://github.com/BR-IDL/PaddleViT.git)，本项目大部分代码都是从中copy来的，而仅仅实现了其与原版代码训练步骤的进一步对齐与完整的训练过程，但本人也同样受益匪浅！:hearts:


## Contact

- Author: Hatimwen

- Email: hatimwen@163.com