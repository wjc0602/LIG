## 风景图片生成赛题b榜代码
图像生成任务一直以来都是十分具有应用场景的计算机视觉任务，从语义分割图生成有意义、高质量的图片仍然存在诸多挑战，如保证生成图片的真实性、清晰程度、多样性、美观性等。

### 部署
1. 运行环境
- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0
- [jimm](https://github.com/Jittor-Image-Models/Jittor-Image-Models)

2. Jimm库使用
- 用于Jittor框架加载预训练模型
- 用到的预训练模型有tf_efficientnet_b5,tf_efficientnet_b6,vit_base_patch16_384,vit_base_patch16_224_in21k,swin_base_patch4_window7_224_in22k

3. Jimm安装(训练需要，测试不需要)，以下给出两种安装方式
3.1 指定目录安装
- 下载开源链接到与train.py文件同目录，解压项目名称命为jimm

3.2 系统安装
- version.py及setup.py存在与train.py代码文件同目录
- 下载开源链接到与train.py文件同目录，解压项目名称命为jimm
- python setup.py install
- 成功后可删除目录下version.py、setup.py、jimm文件

4. requirements.txt文件
- 文件中指定了相应的版本，但不是必须

5. 训练准备及脚本参考命令
- 实验室显卡配置为四卡3090Ti, 训练时间为4~5天
- CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --input_path /data/comptition/jittor/data/train --output_path ./results/
- 显存小于20G, batch_size需调小为2或1

6. 测试脚本模型权重准备及脚本参考命令
- 3个模型权重在model文件夹下
- CUDA_VISIBLE_DEVICES=0,1 python test.py --input_path /data/comptition/jittor/data/val_B-labels-clean --model_path ./model/ --output_path ./results/