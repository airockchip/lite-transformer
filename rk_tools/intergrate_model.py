import math
import os
import numpy as np
import onnxruntime

import torch
import torch.nn as nn
import torch.nn.functional as F

encoder_embed_dim = 256
decoder_embed_dim = 256

max_text_length_for_rknn = 16
bsz = 1


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class EncoderOnnx(torch.nn.Module):
    def __init__(self, model, onnx_model=None, rknn_model=None, target=None):
        super(EncoderOnnx, self).__init__()

        self.embed_dim = encoder_embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.embed_tokens = model.encoder.embed_tokens
        self.embed_positions = model.encoder.embed_positions
        self.layers = model.encoder.layers
        if rknn_model is not None and target is not None:
            from rk_tools.excuter.rknn_excute import RKNN_model_container
            self.rknn_excuter = RKNN_model_container(rknn_model, target)

        if onnx_model is not None:
            from rk_tools.excuter.onnx_excute import ONNX_model_container
            self.onnx_excuter = ONNX_model_container(onnx_model)

    def preprocess(self, src_tokens):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        return x

    def rk_forward(self, x, encoder_pad_mask):
        inputs = [x, encoder_pad_mask]
        inputs = [to_numpy(_d) for _d in inputs]
        out = self.rknn_excuter.run(inputs)
        out = [torch.tensor(_o) for _o in out]
        return out[0]

    def onnx_forward(self, x, encoder_pad_mask):
        inputs = [x, encoder_pad_mask]
        inputs = [to_numpy(_d) for _d in inputs]
        out = self.onnx_excuter.run(inputs)
        out = [torch.tensor(_o) for _o in out]
        return out[0]

    def forward(self, x, encoder_pad_mask):
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, encoder_pad_mask)
        return x


class DecoderOnnx(torch.nn.Module):
    def __init__(self, model, onnx_model=None, rknn_model=None, target=None):
        super(DecoderOnnx, self).__init__()

        self.embed_dim = decoder_embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.embed_tokens = model.decoder.embed_tokens
        self.embed_positions = model.decoder.embed_positions
        self.layers = model.decoder.layers
        self.incremental_state = {}
        self.num_heads = model.decoder.layers[0].self_attn.num_heads
        self.head_dim = model.decoder.layers[0].self_attn.head_dim
        print('num heads: {}\nhead_dim: {}\n'.format(self.num_heads, self.head_dim))

        if rknn_model is not None and target is not None:
            from rk_tools.excuter.rknn_excute import RKNN_model_container
            self.rknn_excuter = RKNN_model_container(rknn_model, target)

        if onnx_model is not None:
            from rk_tools.excuter.onnx_excute import ONNX_model_container
            self.onnx_excuter = ONNX_model_container(onnx_model)

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

    def rk_forward(self, x, encoder_out=None, encoder_pad_mask=None, decoder_pad_mask=None, incremental_states=None):
        inputs = [x, encoder_out, encoder_pad_mask, decoder_pad_mask, *incremental_states]
        inputs = [_d.detach().numpy() for _d in inputs]
        inputs[1] = inputs[1].reshape(1,*inputs[1].shape)
        out = self.rknn_excuter.run(inputs, ['nchw']*len(inputs))
        out = [torch.tensor(_o) for _o in out]
        out = [out[0], out[1:]]
        return out

    def onnx_forward(self, x, encoder_out=None, encoder_pad_mask=None, decoder_pad_mask=None, incremental_states=None):
        inputs = [x, encoder_out, encoder_pad_mask, decoder_pad_mask, *incremental_states]
        inputs = [_d.detach().numpy() for _d in inputs]
        out = self.onnx_excuter.run(inputs)
        out = [torch.tensor(_o) for _o in out]
        out = [out[0], out[1:]]
        return out

    def forward(self, x, encoder_out=None, encoder_pad_mask=None, decoder_pad_mask=None, incremental_states=None, attn_mask=None, src_length=None):
        self_attn_states = []
        x = x.transpose(0, 1)
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                # incremental method
                if incremental_states is not None:
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

                # fix lenght method
                else:
                    x[:-src_length,:,:] = 0
                    x, attn = layer(
                        x,
                        encoder_out=encoder_out,
                        encoder_padding_mask=encoder_pad_mask,
                        self_attn_padding_mask=decoder_pad_mask,
                        prev_self_attn_state=None,
                        self_attn_mask=attn_mask,
                    )
                    self_attn_states = (None, None)

        x = x.transpose(0, 1)
        x = F.linear(x, self.embed_tokens.weight)

        return x, self_attn_states


class Lite_transformer_intergrate_model:
    def __init__(self, encoder, decoder, src_dict, tgt_dict, length_fix=6, incremental_enhance=True) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.length_fix = length_fix
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.v_inf = -65000
        self.incremental_enhance = incremental_enhance
        self.support_framework = ['pt', 'pytorch', 'torch', 'onnx', 'rknn']


    def translate(self, input_token, framework='pt', export_model=False):
        input_tensors = []
        input_tensor_name = []
        output_tensors = []
        output_tensor_name = []

        framework = framework.lower()
        assert framework in self.support_framework, "only support {}, BUT GOT {}".format(self.support_framework, framework)

        # pad to fix length
        real_length = len(input_token)
        if real_length > self.length_fix:
            input_token = input_token[:self.length_fix]
            real_length = self.length_fix
        else:
            # input_token = input_token + [self.tgt_dict.pad()]*(self.length_fix - real_length)     # right pad
            input_token = [self.tgt_dict.pad()]*(self.length_fix - real_length) + input_token       # left pad

        # generate padding mask, order from left to right
        input_token = torch.tensor(input_token).reshape(1, -1)
        encoder_padding_mask = torch.zeros((1, self.length_fix)).float()
        # encoder_padding_mask[:,real_length:]=1.                                                   # right pad
        encoder_padding_mask[:,:-real_length]=1.                                                    # left pad

        # run encoder
        # print('input token: {}'.format(input_token))
        # print('input padding mask: {}'.format(encoder_padding_mask))
        embedded_token = self.encoder.preprocess(input_token)

        if framework == 'rknn':
            encoder_output = self.encoder.rk_forward(embedded_token, encoder_padding_mask.repeat(self.length_fix, 1).unsqueeze(0))
        elif framework == 'onnx':
            encoder_output = self.encoder.onnx_forward(embedded_token, encoder_padding_mask.repeat(self.length_fix, 1).unsqueeze(0))
        else:
            encoder_output = self.encoder(embedded_token, encoder_padding_mask.repeat(self.length_fix, 1).unsqueeze(0))

        # init decoder input token with eos
        decoder_output_tokens = []
        decoder_output_tokens.append(self.tgt_dict.eos())
        score_list = []

        for i in range(self.length_fix):
            # break if meet eos symbol
            if i!=0 and decoder_output_tokens[-1]==self.tgt_dict.eos:
                break
            
            # generate decoder padding mask, order from right to left
            decoder_padding_mask = torch.ones((1, self.length_fix))
            decoder_padding_mask[:,-(i+1):] = 0
            # print('epoch:', i+1)

            if self.incremental_enhance is True:
                if i==0:
                    token_embed, incremental_state = self.decoder.preprocess(torch.tensor(decoder_output_tokens[:]).reshape(1,-1))
                else:
                    token_embed, _ = self.decoder.preprocess(torch.tensor(decoder_output_tokens[:]).reshape(1,-1))
                token_embed = token_embed[:,-1:,:]

                if framework == 'rknn':
                    decoder_output = self.decoder.rk_forward(token_embed, encoder_output, encoder_padding_mask, decoder_padding_mask, incremental_state)
                elif framework == 'onnx':
                    decoder_output = self.decoder.onnx_forward(token_embed, encoder_output, encoder_padding_mask, decoder_padding_mask, incremental_state)
                else:
                    decoder_output = self.decoder(token_embed, encoder_output, encoder_padding_mask, decoder_padding_mask, incremental_state)
            else:
                # incremental_enhance set False only support pytorch
                new_attn_mask = torch.ones(self.length_fix, self.length_fix)* self.v_inf
                new_attn_mask = torch.tril(new_attn_mask, -1).T
                new_attn_mask[:,:-len(decoder_output_tokens[:])]= self.v_inf
                lt = (self.length_fix -len(decoder_output_tokens[:]))*[1] + decoder_output_tokens[:]

                token_embed, _ = self.decoder.preprocess(torch.tensor(lt).reshape(1,-1))
                decoder_output = self.decoder(token_embed, encoder_output, encoder_padding_mask, decoder_padding_mask, 
                                attn_mask=new_attn_mask, src_length=len(decoder_output_tokens[:]))

            # update incremental_state until final time
            if i+1<self.length_fix and self.incremental_enhance:
                for j in range(len(incremental_state)):
                    incremental_state[j] = decoder_output[1][j][:,:,1:,:]
    
            dout = torch.log_softmax(decoder_output[0], -1)
            decoder_output_tokens.append(int(decoder_output[0][:,-1,:].argmax()))
            score_list.append( float( decoder_output[0].reshape(-1)[decoder_output_tokens[-1]]))
            
            # break if meet eos symbol
            if decoder_output_tokens[-1] == self.tgt_dict.eos():
                break

        # print(decoder_output_tokens, '\n')
        # print(score_list, '\n')

        zh_src = [self.tgt_dict.symbols[_t] for _t in decoder_output_tokens]
        zh_merge = []
        merge_with_last_word = False
        for _z in zh_src:
            if _z == '</s>':
                continue

            _z_in = _z[:-2] if _z.endswith('@@') else _z
            if merge_with_last_word is False:
                zh_merge.append(_z_in)
            elif merge_with_last_word is True:
                zh_merge[-1] = zh_merge[-1] + _z_in

            merge_with_last_word = True if _z.endswith('@@') else False

        print(' '.join(zh_merge))
        return 


    def export_model(self, path_dir, inputs_token=None):
        if inputs_token is None:
            input_token = np.random.randint(0, len(self.tgt_dict.indices), (1, self.length_fix))
        
