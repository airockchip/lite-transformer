###
 # Copyright (c) 2022 by Rockchip Electronics Co., Ltd. All Rights Reserved.
 # 
 # 
 # @Author: Randall Zhuo
 # @Date: 2022-09-16 11:25:37
 # @LastEditors: Randall
 # @LastEditTime: 2022-09-16 11:26:58
 #

import math
import os
import sys
import numpy as np
import onnxruntime

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import tasks, search, utils
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils

encoder_embed_dim = 256
decoder_embed_dim = 256

bsz = 1
beam_size = 1
max_text_length_for_rknn = 16
max_len = 200

class EncoderOnnx(torch.nn.Module):
    def __init__(self, model):
        super(EncoderOnnx, self).__init__()

        self.embed_dim = encoder_embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.embed_tokens = model.encoder.embed_tokens
        self.embed_positions = model.encoder.embed_positions
        self.layers = model.encoder.layers

    def preprocess(self, src_tokens):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        return x

    def forward(self, x, encoder_pad_mask):
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, encoder_pad_mask)

        x = x.transpose(0, 1)
        return x


class DecoderOnnx(torch.nn.Module):
    def __init__(self, model):
        super(DecoderOnnx, self).__init__()

        self.embed_dim = decoder_embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.embed_tokens = model.decoder.embed_tokens
        self.embed_positions = model.decoder.embed_positions
        self.layers = model.decoder.layers
        self.incremental_state = {}
        self.num_heads = model.decoder.layers[0].self_attn.num_heads
        self.head_dim = model.decoder.layers[0].self_attn.head_dim
        print('num heads: {}\nhead_dim: {}'.format(self.num_heads, self.head_dim))

    def preprocess(self, prev_output_tokens):
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += self.embed_positions(prev_output_tokens)

        incremental_state = []

        for layer in self.layers:
            prev_key = torch.zeros(bsz, self.num_heads, max_text_length_for_rknn-1, self.head_dim)
            prev_value = torch.zeros(bsz, self.num_heads, max_text_length_for_rknn-1, self.head_dim)
            incremental_state.append(prev_key)
            incremental_state.append(prev_value)
        return x, incremental_state

    def forward(self, x, encoder_out=None, encoder_pad_mask=None, decoder_pad_mask=None, incremental_states=None):
        self_attn_states = []
        x = x.transpose(0, 1)
        encoder_out = encoder_out.transpose(0, 1)
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                prev_key, prev_value = incremental_states[i*2], incremental_states[i*2+1]
                prev_self_attn_state = prev_key, prev_value
                x, attn, self_attn_state = layer(
                    x,
                    encoder_out=encoder_out,
                    encoder_padding_mask=encoder_pad_mask,
                    self_attn_padding_mask=decoder_pad_mask,
                    prev_self_attn_state=prev_self_attn_state,
                )

                save_key, save_value = self_attn_state
                self_attn_states.append(save_key)
                self_attn_states.append(save_value)

        x = x.transpose(0, 1)
        x = F.linear(x, self.embed_tokens.weight)

        return x, self_attn_states

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def torch2onnx(model, src_dict, tgt_dict):
    # input_text = "Hello world."

    # encoder input
    # src_tokens_idx = src_dict.encode_line(input_text).long()

    # only for trace onnx
    src_tokens_idx = np.random.randint(10, 100, (max_text_length_for_rknn))
    src_tokens_idx = np.array(src_tokens_idx)

    src_tokens = torch.tensor(src_tokens_idx, dtype=torch.long).unsqueeze(0)

    encoder_padding_mask = src_tokens.eq(1).float()
    encoder_padding_mask = encoder_padding_mask.repeat(max_text_length_for_rknn,1).unsqueeze(0)

    # encoder forward
    encoder = EncoderOnnx(model)

    embed_src_tokens = encoder.preprocess(src_tokens)

    encoder_out = encoder.forward(embed_src_tokens, encoder_padding_mask)

    np.save("export_onnx/encoder_input.npy", to_numpy(embed_src_tokens))
    np.save("export_onnx/encoder_padding_mask.npy", to_numpy(encoder_padding_mask))
    np.save("export_onnx/encoder_output.npy", to_numpy(encoder_out))
    to_numpy(encoder.embed_tokens.weight).tofile('export_onnx/token_embed.bin')
    to_numpy(encoder.embed_positions.weights).tofile('export_onnx/position_embed.bin')

    encoder_onnx = 'export_onnx/lite-transformer-encoder-{}.onnx'.format(max_text_length_for_rknn)

    torch.onnx.export(encoder, (embed_src_tokens, encoder_padding_mask),
                    encoder_onnx,
                    input_names = ['x', "encoder_padding_mask"],
                    output_names = ['encoder_output'],
                    opset_version=12,
                    export_params=True,
                    do_constant_folding=True)
    
    sess_options  = onnxruntime.SessionOptions()
    ort_session = onnxruntime.InferenceSession(encoder_onnx,sess_options)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(embed_src_tokens), 
                 ort_session.get_inputs()[1].name: to_numpy(encoder_padding_mask)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(encoder_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("{} has been tested with onnxruntime, and the result looks good!".format(encoder_onnx))

    # decoder forward
    decoder = DecoderOnnx(model)

    # compute the encoder output for each beam
    new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
    new_order = new_order.to(src_tokens.device).long()

    # reorder encoder output
    encoder_out = encoder_out.index_select(0, new_order)

    # compute the finally decoder output
    pad = tgt_dict.pad()
    eos = tgt_dict.eos()
    tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(pad)
    tokens[:, 0] = eos

    prev_output_tokens = tokens[:, :max_text_length_for_rknn]

    if decoder.incremental_state is not None:
        prev_output_tokens = prev_output_tokens[:, -1:]

    prev_output_embed_tokens, incremental_state = decoder.preprocess(prev_output_tokens)

    step = 1

    if decoder.incremental_state is None:
        decoder_pading_mask = torch.ones(max_text_length_for_rknn, max_text_length_for_rknn)
        decoder_pading_mask[:step+1,:step+1] = torch.triu(decoder_pading_mask[:step+1,:step+1],diagonal=1)
        decoder_pading_mask = decoder_pading_mask.unsqueeze(0)
    else:
        decoder_pading_mask = torch.ones(max_text_length_for_rknn - step - 1)
        decoder_pading_mask = torch.cat((decoder_pading_mask,torch.zeros(step + 1)),dim = 0).float()
        decoder_pading_mask = decoder_pading_mask.unsqueeze(0)

        encoder_padding_mask = encoder_padding_mask.squeeze()[:1]


    print("decoder_pading_mask:", decoder_pading_mask.size())

    decoder_out = decoder.forward(prev_output_embed_tokens, encoder_out, encoder_padding_mask, decoder_pading_mask, incremental_state)

    np.save("export_onnx/prev_output_tokens.npy", to_numpy(prev_output_embed_tokens))
    np.save("export_onnx/encoder_output.npy", to_numpy(encoder_out))
    np.save("export_onnx/decoder_encoder_padding_mask.npy", to_numpy(encoder_padding_mask))
    np.save("export_onnx/decoder_pading_mask.npy", to_numpy(decoder_pading_mask))
    np.save("export_onnx/decoder_output.npy", to_numpy(decoder_out[0]))
    np.save("export_onnx/prev_key0.npy", to_numpy(decoder_out[1][0][:,:,1:,:]))
    np.save("export_onnx/prev_value0.npy", to_numpy(decoder_out[1][1][:,:,1:,:]))
    np.save("export_onnx/prev_key1.npy", to_numpy(decoder_out[1][2][:, :,1:,:]))
    np.save("export_onnx/prev_value1.npy", to_numpy(decoder_out[1][3][:, :,1:,:]))
    np.save("export_onnx/prev_key2.npy", to_numpy(decoder_out[1][4][:, :,1:,:]))
    np.save("export_onnx/prev_value2.npy", to_numpy(decoder_out[1][5][:, :,1:,:]))

    decoder_onnx = 'export_onnx/lite-transformer-decoder-{}.onnx'.format(max_text_length_for_rknn)

    torch.onnx.export(decoder, (prev_output_embed_tokens, encoder_out, encoder_padding_mask, decoder_pading_mask, incremental_state),
                    decoder_onnx,
                    input_names = ['prev_output_tokens', 'encoder_out', 'encoder_padding_mask', 'decoder_pading_mask', 
                                    "prev_key0", "prev_value0", "prev_key1", "prev_value1", "prev_key2", "prev_value2"],
                    output_names = ['decoder_output', "save_key0", "save_value0", "save_key1", "save_value1", "save_key2", "save_value2"],
                    opset_version=12,
                    export_params=True,
                    do_constant_folding=True)    

    sess_options  = onnxruntime.SessionOptions()
    ort_session = onnxruntime.InferenceSession(decoder_onnx,sess_options)

    ort_inputs = {  ort_session.get_inputs()[0].name: to_numpy(prev_output_embed_tokens), 
                    ort_session.get_inputs()[1].name: to_numpy(encoder_out),
                    ort_session.get_inputs()[2].name: to_numpy(encoder_padding_mask),
                    ort_session.get_inputs()[3].name: to_numpy(decoder_pading_mask),
                    ort_session.get_inputs()[4].name: to_numpy(incremental_state[0]),
                    ort_session.get_inputs()[5].name: to_numpy(incremental_state[1]),
                    ort_session.get_inputs()[6].name: to_numpy(incremental_state[2]),
                    ort_session.get_inputs()[7].name: to_numpy(incremental_state[3]),
                    ort_session.get_inputs()[8].name: to_numpy(incremental_state[4]),
                    ort_session.get_inputs()[9].name: to_numpy(incremental_state[5]),
                }
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(decoder_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("{} has been tested with onnxruntime, and the result looks good!".format(decoder_onnx))


def main():
    model_file = sys.argv[1]

    if not os.path.exists("export_onnx"):
        os.mkdir("export_onnx")

    print('lite-transformer model loading from {}'.format(model_file))

    state = torch.load(model_file, map_location=torch.device('cpu'))
    args = state['args']
    args.data = os.path.dirname(model_file)           # if the dataset was moved, this line could help.
    task = tasks.setup_task(args)

    model = task.build_model(args)
    model.load_state_dict(state['model'], strict=False)
    model.prepare_for_onnx_export_()
    model.eval()

    torch2onnx(model, task.source_dictionary, task.target_dictionary)
    exit(0)


if __name__ == '__main__':
    main()
