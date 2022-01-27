# WzzClas
This toolbox involve **Metric learning**, also support **Classifacation**, **Openset Retrieval** task.

## 模型的训练、测试方法说明
### 1.构造数据集
首先按照如下方式构建数据集。注意，每一个类别需要至少包含**2**张图片。

单个的Dataset folder should be like:

Dataset_A

    --Cls_0:
        -xxx.jpg
        -xxx.jpg
        ...
    --Cls_1:
        -xxx.jpg
        -xxx.jpg
        ...
    ...
如果需要同时使用多个Dataset的组合，则每个Dataset都按照上述规则构建，再通过prepare_triplet_data.py进行聚合。  
使用prepare_triplet_data.py脚本，参数说明如下：
```python
python tools/prepare_triplet_data.py
    --folder_path \ # 训练集路径，若有多个数据集，则依次输入即可，Like：--folder_path Dataset_A Dataset_B Dataset_C
    --target_dir \ # txt存放路径
    --test_partion \ # 测试集占比
```

### 2.训练模型  
#### 2.1. 训练：embedding  
使用train_embedding.py脚本，参数说明如下：
```python
python train_embedding.py 
    --train_txt \ # 训练集txt
    --val_txt \ # 测试集txt
    --test_template \ # 测试集template图路径
    --test_query \ # 测试集query图路径
    --save_dir xxx \ # 存放训练好的模型权重文件、训练日志、模型config
    --batch-size \ # 训练和验证时的batch size，默认为16
    --model_name \ # 模型名, 目前只支持EmbeddingModel
    --metrics \ # 模型选用的度量loss，默认为Triplet Loss
    --epochs \ # 训练的总epoch数，默认为100
    --eval_period \ # 训练时的验证步数，默认为5，意味着每5个epoch都进行一次验证
    --earlystop_period \ # 提前停止策略步数，默认为0意味着不会提前终止，若大于0，则意味着如果n次验证得到的模型的准确率没有提升的话训练过程会提前终止
    --lr \ # 学习率，默认为0.002
    --num_workers \ # 默认为2
    --resume \ # 可选项，是否继续继续训练，加载已经训好的模型和学习率以及当前的epoch等参数。（模型意外终止可用它重新启动训练）
    --pretrained_model \ # resume下，预训练模型路径
    --pretrained_config \ # resume下，预训练模型cofig
    --gpu \ # 选用gpu的id
```
#### 2.2. 训练：classifier  
使用train_cls.py脚本，参数说明如下：
```python
python train_cls.py 
    --train_txt \ # 训练集txt
    --val_txt \ # 测试集txt
    --test_template \ # 测试集template图路径
    --test_query \ # 测试集query图路径
    --save_dir xxx \ # 存放训练好的模型权重文件、训练日志、模型config
    --batch-size \ # 训练和验证时的batch size，默认为16
    --model_name \ # 模型名, 目前只支持EmbeddingModel
    --metrics \ # 模型选用的度量loss，默认为Triplet Loss
    --epochs \ # 训练的总epoch数，默认为100
    --eval_period \ # 训练时的验证步数，默认为5，意味着每5个epoch都进行一次验证
    --earlystop_period \ # 提前停止策略步数，默认为0意味着不会提前终止，若大于0，则意味着如果n次验证得到的模型的准确率没有提升的话训练过程会提前终止
    --lr \ # 学习率，默认为0.002
    --num_workers \ # 默认为2
    --resume \ # 可选项，是否继续继续训练，加载已经训好的模型和学习率以及当前的epoch等参数。（模型意外终止可用它重新启动训练）
    --pretrained_model \ # resume下，预训练模型路径
    --pretrained_config \ # resume下，预训练模型cofig
    --gpu \ # 选用gpu的id
```
#### 2.3. DP Mode Example：  
```python
nohup python -u train_embedding.py --train_txt train.txt --val_txt val.txt --resume True --pretrained_model xx --pretrained_config xx --gpu '0,1' > train_log/training.log 2>&1 &
```  

#### 2.4. DDP Mode Example：  
```python
nohup python -u -m torch.distributed.launch --nproc_per_node=4 train_embedding_ddp.py --ddp True > train_log/training.log 2>&1 &
```  

### 3.测试模型
使用test_embedding.py脚本，参数说明如下：
```python
python test_embedding.py 
    --template_dir \ # 测试集标准图路径
    --close_dir \ # 测试集query图路径
    --open_dir \ # 测试集开集图路径
    --save_dir xxx \ # 存放训练好的模型权重文件
    --batch-size \ # 训练和验证时的batch size，默认为224
    --model_name \ # 模型名, 目前只支持EmbeddingModel
    --pretrained-model \ # 预训练模型路径
    --num_workers \ # 默认为2
    --gpu \ # 选用gpu的id
```

### 4.Inference
使用match_infer.py脚本，参数说明如下：
```python
python train_embedding.py 
    --config \ # 模型cfg
    --model_weight \ # 模型路径
    --image_0 \ # 测试图1
    --image_1 \ # 测试图2
```