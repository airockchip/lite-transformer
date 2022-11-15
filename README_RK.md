# Lite Transformer

本工程只是为了展示如何训练及导出适合RK NPU平台的transformer模型。采取的dataset为news-commentary-v15，英文转中文的新闻数据集，数据集本身不大，因此英文转中文的实际效果需要改进。

为了提高decoder的效率，在导出onnx时，使用的是增量解码（incremental decoding）的模型，增量解码可以极大的减少decoder的耗时。具体原理参考：

http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq

由于NPU不支持动态shape输入，通常情况下为了减少计算量，需要导出不同shape的模型，这对于内存、存储都是比较大的浪费。采用增量解码后，这个问题可以得到有效的改善，因为每次送给解码模型都只有一个decoder embeding的长度，也就不存在动态shape的问题。





## 环境安装

### 安装lite-transformer

```
mkdir -p ~/nmt/
cd ~/nmt/
git clone https://github.com/airockchip/lite-transformer.git
cd lite-transformer
pip install --editable .
```



## 数据预处理

这部分主要参考：https://blog.csdn.net/qq_42734797/article/details/112916511#t23

### 依赖包安装

```
mkdir -p ~/nmt/tools
cd  ~/nmt/tools
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git
pip install jieba
```

### 下载数据集

```
mkdir -p ~/nmt/dataset/ncv15/
cd ~/nmt/dataset/ncv15/
wget https://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-zh.tsv.gz
```
并解压出news-commentary-v15.en-zh.tsv

### 预处理
根据实际情况修改lite-transformer/rk_tools/dataset_preprocess.sh中source的路径，以及lite-transformer/rk_tools/preprocess_env.sh中NMT_ROOT的路径
执行
```
lite-transformer/rk_tools/dataset_preprocess.sh
```
进行数据预处理。 



## 训练

执行如下命令进行训练

```
CUDA_VISIBLE_DEVICES="0,1,2" python3 train.py ~/nmt/dataset/ncv15/ --configs configs/rk_nmt.yml
```

可以根据需要修改configs/rk_nmt.yml中的max-update来决定迭代的epoch。



## 导出ONNX模型

```
修改 ./rk_tools/export_onnx.sh 中 model_file 路径
执行 ./rk_tools/export_onnx.sh 在export_onnx目录下导出encoder/decoder的onnx模型。

注意:
Pytorch版本建议使用 1.11 以上版本，1.10 版本在转出 onnx 模型时可能会遇到问题
```

相关模型权重可从[百度网盘](https://eyun.baidu.com/s/3humTUNq)获取，密码为 rknn



## 翻译demo

```
修改 ./rk_tools/translate_demo.sh 中，修改:
	bpe_dict 		分词字典路径，该文件根据训练语料库生成
	bpe_apply_file 	分词执行文件，指向 subword-nmt 仓库的 apply_bpe.py 文件
	model_file 		指向模型路径
	framework 		推理框架，支持pt/onnx/rknn
执行 translate_demo.sh
```

