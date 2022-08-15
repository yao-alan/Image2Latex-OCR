import torch
from torch import nn
from torch.nn import functional as F

class CRNN(nn.Module):
    """CNN encoder, GRU decoder."""
    def __init__(self, vocab=None, embed_size=128, hidden_size=256, **kwargs):
        super(CRNN, self).__init__(**kwargs)
        # CNN structure from HarvardNLP paper
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1)
        )
        if vocab is not None:
            # embedding to be appended to CNN output
            self.embedding = nn.Embedding(
                num_embeddings=len(vocab), 
                embedding_dim=embed_size,
                dtype=torch.float32
            )
            # bidirectional, 2-layer GRU
            self.rnn = nn.GRU(
                input_size=512+embed_size, 
                hidden_size=hidden_size, 
                num_layers=2,
                dropout=0.3, 
                bidirectional=True,
                batch_first=True
            )
            self.dense = nn.Linear(
                in_features=2*hidden_size,
                out_features=len(vocab)
            )
            self.vocab = vocab
            
            self.embed_size = embed_size
            self.hidden_size = hidden_size


    def forward(self, X, label=None, teacher_forcing=False):
        X = self.cnn(X / 255)
        # X.shape = (batch_size, feature_size (n_channels), height, width)
        X = X.contiguous().view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])
        assert len(X.shape) == 3
        # X.shape = (batch_size, feature_size (n_channels), seq_len (width))
        if self.training and teacher_forcing:
                # create embedding vectors from label
                # label.shape = (batch_size, seq_len)
                assert label is not None
                label = label[:, :-1].long()
                # initial embed_tensor.shape = (batch_size, seq_len, embed_dim)
                embed_tensor = self.embedding(label).permute(0, 2, 1)
                # embed_tensor.shape = (batch_size, embed_dim, seq_len)
                # initial X.shape = (batch_size, feature_size+embed_dim, seq_len)
                X = torch.cat([X, embed_tensor], dim=1).permute(0, 2, 1)
                # X.shape = (batch_size, seq_len, feature_size+embed_dim)
                Y, state = self.rnn(X)
                # Y.shape = (batch_size, seq_len, 2 * hidden_size)
                # if using CTC loss
                Y = self.dense(Y).permute(1, 0, 2)
                # Y.shape = (seq_len, batch_size, vocab_size)
                return Y
        else:
            seq = torch.ones(
                (X.shape[0], X.shape[2]), dtype=torch.long
            ).to(X.device) * self.vocab.get_index('<beg>')
            # Y.shape = (seq_len, batch_size, vocab_size)
            Y = torch.ones((X.shape[2], X.shape[0], len(self.vocab))).to(X.device)

            state = torch.zeros((2 * 2, seq.shape[0], self.hidden_size)).to(X.device)
            for t in range(1, seq.shape[1], 1):
                # curr_label.shape = (batch_size, 1) -> labels at time t-1
                curr_label = seq[:, t-1:t].clone() # use t-1:t to keep dims
                embed_tensor = self.embedding(curr_label).permute(0, 2, 1)
                # embed_tensor.shape = (batch_size, embed_dim, seq_len)
                input_tensor = torch.cat(
                    [X[:, :, t-1:t], embed_tensor], dim=1
                ).permute(0, 2, 1)

                out, state = self.rnn(input_tensor, state)
                # out.shape = (batch_size, 1, 2 * hidden_size)
                out = self.dense(out).permute(1, 0, 2)
                # out.shape = (1, batch_size, vocab_size)
                Y[t] = out
                out = F.softmax(out, dim=2)
                seq[:, t] = out.argmax(dim=2)

            if self.training:
                return Y
            else:
                return seq


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.GRU):
        for p in m._flat_weights_names:
            if 'weight' in p:
                nn.init.xavier_uniform_(m._parameters[p])