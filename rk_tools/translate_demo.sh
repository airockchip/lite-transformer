bpe_dict=./pretrained/ncv15/bpe.txt
bpe_apply_file=../tools/subword-nmt/apply_bpe.py
model_file=./pretrained/ncv15/checkpoint_best.pt

# pt, onnx, rknn 
# if onnx was using, export onnx model first. refer to 'rk_tools/export_onnx.sh'
# if rknn was using, sign the rknn_path and target_platfrom in the 'rk_tools/translate_demo.py'
framework=pt

python3 rk_tools/translate_demo.py $model_file $bpe_dict $bpe_apply_file $framework
