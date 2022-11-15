import os
import sys
import torch

from rk_tools.intergrate_model import EncoderOnnx, DecoderOnnx, Lite_transformer_intergrate_model
from fairseq import tasks, search, utils
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def words_to_token(words, src_dict):
    if isinstance(words, str):
        words = words.split(" ")
    if not isinstance(words, list):
        return False

    output_token = []
    for _w in words:
        # all unfind word set token as 3
        _tk = src_dict.indices.get(_w, 3)
        if _tk == 3:
            print("  '{}' is not found in dictionary, set as <unk>".format(_w))
        output_token.append(_tk)
    output_token.append(2)
    return output_token


def apply_bpe(words, bpe_apply_file, bpe_dict_path):
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    tmp_file = "./tmp/word.txt"
    tmp_bpe_file = "./tmp/word_bpe.txt"
    if not isinstance(words, str):
        return False
    with open(tmp_file, 'w') as f:
        f.write(words)
    os.system("python {} -c {} < {} > {}".format(bpe_apply_file, bpe_dict_path, tmp_file, tmp_bpe_file))
    
    with open(tmp_bpe_file, 'r') as f:
        bpe_word = f.readline()
    
    return bpe_word


def main():
    model_file = sys.argv[1]
    bpe_dict = sys.argv[2]
    bpe_apply_file = sys.argv[3]
    framework = sys.argv[4]

    if not os.path.exists("export_onnx"):
        os.mkdir("export_onnx")
    print('lite-transformer model loading from {}'.format(model_file))

    state = torch.load(model_file, map_location=torch.device('cpu'))
    args = state['args']
    args.data = os.path.dirname(model_file)             # if the dataset was moved, this line could help.
    task = tasks.setup_task(args)

    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state['model'])
    model.prepare_for_onnx_export_()
    model.eval()

    target_device = 'rk3568'
    rk_enc = None
    rk_dec = None

    # onnx_enc = './export_onnx/lite-transformer-encoder-16.onnx'
    # onnx_dec = './export_onnx/lite-transformer-decoder-16.onnx'
    onnx_enc = None
    onnx_dec = None

    encoder = EncoderOnnx(model, onnx_enc, rk_enc, target_device)
    decoder = DecoderOnnx(model, onnx_dec, rk_dec, target_device)
    
    intergrate_model = Lite_transformer_intergrate_model(
                            encoder, 
                            decoder,
                            task.source_dictionary,
                            task.target_dictionary,
                            length_fix=16,
                            incremental_enhance=True)

    while(1):
        words = input("\n请输入需要翻译的语句,输入q退出:\n")
        if words == "q":
            exit()
        elif words == "":
            continue

        bpe_word = apply_bpe(words, bpe_apply_file, bpe_dict)
        if bpe_word == words:
            print("after bpe, words is unchange.")
        else:
            print("after bpe, words is changed as\n  {}".format(bpe_word))

        input_token = words_to_token(bpe_word, task.source_dictionary)
        print("input token: {}".format(input_token))

        intergrate_model.translate(input_token=input_token,
                                   framework=framework)


if __name__ == '__main__':
    main()