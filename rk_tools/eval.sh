checkpoints_path=$1
data=$2
subset="test"

mkdir -p $checkpoints_path/exp

CUDA_VISIBLE_DEVICES=0 python -W ignore generate.py $data \
        --path "$checkpoints_path/checkpoint_best.pt" --gen-subset $subset \
        --beam 1 --batch-size 1 --remove-bpe  --lenpen 0.6 \
        > $checkpoints_path/exp/${subset}_gen.out 

GEN=$checkpoints_path/exp/${subset}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

grep ^H- $GEN | cut -f3- > $SYS
grep ^T- $GEN | cut -f2- > $REF
python score.py --sys $SYS --ref $REF | tee $checkpoints_path/exp/checkpoint_best.result
