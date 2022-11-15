#!/bin/bash -x

source ~/nmt/lite-transformer/rk_tools/preprocess_env.sh

mkdir -p ${model_dir}

# 将一个单独的数据文件切分成标准格式，即源语言(raw.zh)、目标语言(raw.en)文件各一个，一行一句
python ${utils}/cut2.py ${data_dir}/news-commentary-v15.en-zh.tsv ${data_dir}/

# 标点符号的标准化，同时对双语文件(raw.en, raw.zh)处理
perl ${NORM_PUNC} -l en < ${data_dir}/raw.en > ${data_dir}/norm.en
perl ${NORM_PUNC} -l zh < ${data_dir}/raw.zh > ${data_dir}/norm.zh

# 对标点符号标准化后的中文文件(norm.zh)进行分词处理
python -m jieba -d " " ${data_dir}/norm.zh > ${data_dir}/norm.seg.zh

# tokenize
${TOKENIZER} -l en < ${data_dir}/norm.en > ${data_dir}/norm.tok.en
${TOKENIZER} -l zh < ${data_dir}/norm.seg.zh > ${data_dir}/norm.seg.tok.zh

# 对上述处理后的英文文件(norm.tok.en)进行大小写转换处理(对于句中的每个英文单词，尤其是句首单词
${TRAIN_TC} --model ${model_dir}/truecase-model.en --corpus ${data_dir}/norm.tok.en
${TC} --model ${model_dir}/truecase-model.en < ${data_dir}/norm.tok.en > ${data_dir}/norm.tok.true.en

# 对上述处理后的双语文件(norm.tok.true.en, norm.seg.tok.zh)进行子词处理
python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.tok.true.en  -s 32000 -o ${model_dir}/bpecode.en --write-vocabulary ${model_dir}/voc.en
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.en --vocabulary ${model_dir}/voc.en < ${data_dir}/norm.tok.true.en > ${data_dir}/norm.tok.true.bpe.en

python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.seg.tok.zh  -s 32000 -o ${model_dir}/bpecode.zh --write-vocabulary ${model_dir}/voc.zh
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.zh --vocabulary ${model_dir}/voc.zh < ${data_dir}/norm.seg.tok.zh > ${data_dir}/norm.seg.tok.bpe.zh

#clean
mv ${data_dir}/norm.seg.tok.bpe.zh ${data_dir}/toclean.zh
mv ${data_dir}/norm.tok.true.bpe.en ${data_dir}/toclean.en
${CLEAN} ${data_dir}/toclean zh en ${data_dir}/clean 1 256

# 双语文件(clean.zh, clean.en)都需要按比例划分出训练集、测试集、开发集
python ${utils}/split.py ${data_dir}/clean.zh ${data_dir}/clean.en ${data_dir}/

# 预处理后的六个文件(train.zh, valid.en等)，使用fairseq-preprocess命令生成词表和训练用的二进制文件
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${data_dir}/train --validpref ${data_dir}/valid --testpref ${data_dir}/test \
    --destdir ${data_dir}/data-bin  \
    --joined-dictionary
