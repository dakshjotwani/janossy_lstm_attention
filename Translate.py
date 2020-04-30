import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.data import Dataset

from transformer import Transformer, get_pad_mask, get_subsequent_mask




'''FILE: Translator.py'''
''' This module will handle the text generation with beam search. '''

# from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask

class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)


    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.beam_size

        enc_output, *_ = self.model.encoder(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)
        
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores


    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
    

    
    
    
    
'''FILE: translate.py'''
''' Translate input text with trained model. '''

# from transformer.Models import Transformer
# from transformer.Translator import Translator


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    # TODO: Translate bpe encoded files 
    #parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    #parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    #parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    unk_idx = SRC.vocab.stoi[SRC.unk_token]
    with open(opt.output, 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            #print(' '.join(example.src))
            src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
            pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
            #print(pred_line)
            f.write(pred_line.strip() + '\n')

    print('[Info] Finished.')

if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()