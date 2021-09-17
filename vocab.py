import numpy as np
import torch

class Vocab:
    # if a token appears < min_freq times, give it token <unk>
    #   <unk> can be substituted in with \mathord{?} when rendering
    def __init__(self, token_list=None, min_freq=0):
        self.vocab = {}
        self.vocab_list = [] # vocab ordered by decreasing frequency
        self.index_dict = {}
        self.min_freq = min_freq

        if token_list is not None:
            self.update(token_list)

    def update(self, token_list):
        token_list = np.asarray(token_list)
        assert len(token_list.shape) == 1 or len(token_list.shape) == 2

        if len(token_list.shape) == 1:
            token_list = np.expand_dims(token_list, 0)

        for tokens in token_list:
            for t in tokens:
                if t in self.vocab:
                    self.vocab[t] += 1
                else:
                    self.vocab[t] = 1

        sorted_list = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        # '<nul>' used as CTC loss blank
        self.vocab_list = ['<nul>', '<unk>', '<beg>', '<end>', '<pad>']
        self.index_dict = {'<nul>': 0, '<unk>': 1, '<beg>': 2, '<end>': 3, '<pad>': 4}
        i, orig_len = 0, len(self.vocab_list)
        for token, count in sorted_list:
            if count >= self.min_freq:
                self.vocab_list.append(token)
                self.index_dict[token] = orig_len + i
                i += 1

    def get_freq(self, token):
        if token in self.vocab:
            return self.vocab[token]
        else:
            return 0

    def get_index(self, token):
        if token in self.index_dict:
            return self.index_dict[token]
        else:
            return self.get_index('<unk>')

    def get_token(self, index):
        return self.vocab_list[index]

    def __len__(self):
        return len(self.vocab_list)


def label_to_index(label, vocab, maxlen=200):
    """Convert words/symbols/commands to vocab indices."""
    if maxlen is None:
        maxlen = len(label)
        
    label = label[:maxlen-1]
    indices = torch.empty((maxlen))
    indices[0] = vocab.get_index('<beg>')

    for i in range(len(label)):
        indices[i+1] = vocab.get_index(label[i])
        
    # <end> token and padding
    if len(label) < maxlen-1:
        indices[len(label)+1] = vocab.get_index('<end>')
        indices[len(label)+2 : maxlen] = vocab.get_index('<pad>')

    return indices.tolist()


def indices_to_latex(index_tensor, vocab, using_ctc_loss=True):
    """Convert vocab indices to words/symbols/commands."""
    if using_ctc_loss:
        index_tensor = ~(torch.hstack(
            (index_tensor[:, 1:], torch.zeros((index_tensor.shape[0], 1)
            ).to(index_tensor.device))) == index_tensor) * index_tensor

    latex_phrases = []
    for indices in index_tensor:
        phrase = ""
        ignore = set(['<nul>', '<beg>', '<end>', '<pad>'])
        for x in indices:
            token = vocab.get_token(x)
            if token not in ignore:
                if token == '<unk>':
                    token = '\\mathord{?}'
                phrase += token + " "
            elif token == '<end>':
                phrase += " "
                break
        latex_phrases.append(phrase[:-1])

    return latex_phrases
