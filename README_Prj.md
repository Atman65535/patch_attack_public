# 利用扩散模型的注意力功能加强对抗补丁的隐匿性

## 文件结构：
基于mmsegmentation，新增代码文件都在./usr下，包含：
- train_patch.py 主函数入口

- configs，包含模型配置和训练配置，该文件夹是删除了部分冗余文件的mmseg的configs文件夹内容。另外包含**exp**文件夹，为适配过rellis数据集的模型；**exp/patch_config.py**，是服从mmlab配置风格写的训练配置文件。打开即可查看更多信息；pretrained文件夹下放置预训练模型。
- datasets，为适配rellis写的数据集类。
- diffusion loss neat，第二代diffusion loss代码，包含UNet注入注意力捕捉器unet_patch.py，ddim逆过程产生潜空间张量的ddim_reverse.py，注意力捕捉器attention_catcher.py，和diffusion_utils工具文件。主要的**主程序接口和用户接口是diff_loss_pipelien.py**中实现的图像操控类，输入图像，返回自注意和交叉注意loss。
- patch patch管理类，包含patch_handler.py，主要负责存储和对图像施加补丁。
- metrics 除了diffusionloss以外的loss管理类。仅patch_metrics_v1.py有用，iou并未实现，目前看来对训练没什么帮助，老版本的patch_metrics.py弃用。

- utils 训练工具类。loss_handler本来是想做成一个梯度累积工具，同时打印loss的log，但是尚未成功，用handler类管理梯度容易出现计算图复用的报错，目前没有做完这里。utils.py主要做了MMlab系列语义分割数据结构的拆解，因为mm的模型推理完出来不是裸梯度，需要解包一下。

- adv_attack_plugin，为参考代码
- diffusionattack 第一代diffusion loss，包含了一些高端复用功能，比如说hijack decorator可以将任意功能函数注入某个网络等。**本文件夹的代码均未调试跑通，目前与项目无关，未来可能写成工具类积累起来**

## 当前进展
1. 扩散Pipeline基本完成，每次扩散单张图像，获得自注意和交叉注意损失。  
2. 分类器loss基本完成
3. 总体loss不收敛，可能是训练时间不足，训练一小时左右，loss基本在一个值附近浮动，没有任何下降趋势。

## TODO
- [ ] patch存储功能还没写，计划写个pickle存在patch_config.py中指定的地点。
- [ ] 训练过程可视化。为了解决收敛问题，计划写一个工具类，能够抽取施加补丁之后输入模型的图像，观察patch是否有变化。
- [ ] EOT变换，目前在patch handler里面有一个初版，但是短期不计划启用，如果loss收敛可以考虑开启EOT。开启方式在patch_config.py里面有个关键字。但是EOT没调试过，不清楚情况。这个依赖第一个TODO的可视化。
- [ ] 封装分类器Pipeline，调用mmlab的语义分割API很繁琐，有很多种不同的数据结构来存储预处理数据，推理结果等。如果能封装起来就好了。

---
下面是一些悬而未决的问题
- [ ] self attention的loss低于cross attention 数个数量级，权重给的大小很重要。
- [ ] 目前patch采用平方映射，初始化写成什么值有待商榷。或者改用其他映射将patch的值映射到01区间内。这个可能是训练不收敛的原因之一。
- [ ] 预处理类可能写的不太行，不知道处理之后是个什么效果。mmlab的模型自带预处理器，但是不清楚内部做了什么，可能有过多的padding导致全黑图像，这使得我们可能要重写一个预处理
- [ ] loss 管理还是比较混乱，梯度累积暂时没有调好。
- [ ] patch metric里面l1正则loss还没有写，smooth loss写了没用上

- [ ] 暂时只支持灰度patch，RGBpatch写了一部分，但是写死了如果用RGB会直接raise 错误。灰度可能更容易收敛？
- [ ] patch handler似乎初始化需要loss权重，但是它自己不管理loss。这个代码冲突需要修改。
- [ ] 补丁初始化位置目前只能是中心，可以未来修改。
- [ ] difflosspipeline 里面类名是DiffLossTools，有机会可以改。
- [ ] 目前并行化能力较差，仅支持单图推理。但是单图占用显存也不小。如果要改并行，应该得大动干戈。我的考虑是暂时不加并行多卡，改用梯度累计。

## 版本
pytorch                   1.10.0          py3.8_cuda11.3_cudnn8.2.0_0    
pytorch-mutex             1.0                        cuda    pytorch   
torchaudio                0.10.0               py38_cu113    pytorch   
torchvision               0.11.0               py38_cu113    pytorch   
diffusers                 0.30.3                   pypi_0    pypi   
accelerate                1.0.1                    pypi_0    pypi   
transformers              4.46.3                   pypi_0    pypi   

## 运行方法
1. 在根目录下新建data文件夹，软连接rellis3d数据集到data/rellis3d下，保证数据集里面有.lst索引文件
2. 在usr/configs/exp/patch_config.py中修改相应字段
    1. load_from : 请在对应位置放置预训练权重，目前使用的是bisenetv2.pth
    2. 似乎只有这一个需要改动
3. 直接启动train_patch.py

## 注意事项
classifier 图像要求是01区间，标准化的图像（-mean / std）但是扩散要求是-1 1区间。