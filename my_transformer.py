import copy
import torch
import math
from torch import nn
from torch.nn.functional import log_softmax


# modify Tensor __repr__ method
def custom_repr(self):
    return "{}, {}".format(self.shape, origin_repr(self))

origin_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def show_example(fn, args=[], kwargs={}):
    return fn(*args, **kwargs)

def execute_example(fn, args=[], kwargs={}):
    fn(*args, **kwargs)

def clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        return self.encoder(src, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        return self.decoder(memory, tgt, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

class Generator(nn.Module):
    def __init__(self, n_model, vocab):
        super().__init__()
        self.linear = nn.Linear(n_model, vocab)

    def forward(self, x):
        x = self.linear(x)
        return log_softmax(x, dim=-1)

class Encoder(nn.Module):
    def __init__(self, encoder_layer, N=6):
        super().__init__()
        self.layers = clones(encoder_layer, N)
        self.norm = nn.LayerNorm(encoder_layer.size)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, attn, ffn, dropout):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.size = size
        self.dropout = dropout
        self.sublayers = clones(SubLayerConnection(self.size, self.dropout), 2)

    def forward(self, x, src_mask):
        x = self.sublayers[0](x, lambda x: self.attn(x, x, x, src_mask))
        return self.sublayers[1](x, self.ffn)
    
class Decoder(nn.Module):
    def __init__(self, decoder_layer, N=6):
        super().__init__()
        self.layers = clones(decoder_layer, N)
        self.norm = nn.LayerNorm(self.layers[0].size)

    def forward(self, memory, x, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(memory, x, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, size, src_attn, self_attn, ffn, dropout):
        super().__init__()
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.dropout = dropout
        self.ffn = ffn
        self.size = size
        self.sublayers = clones(SubLayerConnection(self.size, self.dropout), 3)
    
    def forward(self, mem, x, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.src_attn(x, mem, mem, src_mask))
        x = self.sublayers[1](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayers[2](x, self.ffn)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab, d_model)

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=20000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos_en = torch.zeros((max_len, d_model))
        div_term = torch.exp(-math.log(10000) * torch.arange(0, d_model, 2) / d_model)
        pos_en[:, 0::2] = torch.sin(torch.arange(0, max_len).unsqueeze(1) * div_term)
        pos_en[:, 1::2] = torch.cos(torch.arange(0, max_len).unsqueeze(1) * div_term)
        pos_en = pos_en.unsqueeze(0)
        self.register_buffer('pos_en', pos_en)

    def forward(self, x):
        x = x + self.pos_en[:, :x.size(-2)].requires_grad_(False)  # default is false
        return self.dropout(x)
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.linear1(x).relu()))

def attention(query, key, value, mask=None, dropout=None):
    "Implementation of attention layer."
    d_k = query.size(-1)
    score = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask==0, -1e9)
    p_attn = score.softmax(dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    out = torch.matmul(p_attn, value)
    return out, p_attn

def subsequent_mask(n):
    mask = torch.triu(torch.ones(1, n, n), diagonal=1)
    return mask == 0

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert self.d_model % self.n_head == 0
        self.dk = self.d_model // self.n_head
        self.dropout = nn.Dropout(dropout)
        self.linear = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, q, k, v, mask=None):
        # unqueeze to apply to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = q.size(0)
        q, k, v = [
            lin(x).view(batch_size, -1, self.n_head, self.dk).transpose(1, 2)
             for x, lin in zip([q, k, v], self.linear)
        ]

        x, self.attn = attention(q, k, v, mask, self.dropout)

        x = x.transpose(1, 2) \
                .reshape(batch_size, -1, self.d_model) \
                .contiguous()
        
        x = self.linear[-1](x)
        del q
        del k
        del v
        return x

def make_model(
        src_vocab,
        tgt_vocab,
        N=6,
        d_model=512,
        d_ffn=2048,
        n_head=8,
        dropout=0.1
) -> EncoderDecoder :
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_head, d_model, dropout)
    ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding(d_model, dropout))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), PositionalEncoding(d_model, dropout))
    generator = Generator(d_model, tgt_vocab)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ffn), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ffn), dropout), N),
        c(src_embed),
        c(tgt_embed),
        c(generator)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def inference():
    src_vocab, tgt_vocab = 15, 15
    model = make_model(src_vocab, tgt_vocab, 6)
    model.eval()
    src = torch.arange(0, 10, dtype=torch.long).unsqueeze(0)
    src_mask = torch.ones(1, 1, 10)
    ys = torch.zeros(1, 1).type_as(src)
    for _ in range(10):
        out = model(src, ys, src_mask, subsequent_mask(ys.size(-1)))
        prob = model.generator(out[:, -1])
        _, next = torch.max(prob, dim=-1)
        ys = torch.concat((ys, torch.ones(1, 1).fill_(next.detach()[0]).type_as(src)), -1)
    print('Input: {} Result: {}'.format(src, ys))

def run_test():
    for _ in range(10):
        inference()

# run_test()

'''
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class Ori_EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

class Ori_Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
    

def Ori_clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Ori_Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class Ori_SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    

class Ori_EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    

class Ori_Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

class Ori_DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    

def Ori_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Ori_MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
    

class Ori_PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    

class Ori_Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000) / d_model)
        pe[:, 0::2] = torch.sin(torch.arange(0, max_len).unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(torch.arange(0, max_len).unsqueeze(1) * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class My_PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=20000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos_en = torch.zeros((max_len, d_model))
        div_term = torch.exp(-math.log(10000) * torch.arange(0, d_model, 2) / -d_model)
        pos_en[:, 0::2] = torch.sin(torch.arange(0, max_len).unsqueeze(1) * div_term)
        pos_en[:, 1::2] = torch.cos(torch.arange(0, max_len).unsqueeze(1) * div_term)
        pos_en = pos_en.unsqueeze(0)
        self.register_buffer('pos_en', pos_en)

    def forward(self, x):
        x = x + self.pos_en[:, :x.size(-2)].requires_grad_(False)  # default is false
        return self.dropout(x)

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
'''

# ========================= Training =========================
# ============================================================
# ============================================================

from dataclasses import dataclass
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
import time

'''
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 0 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
    

def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# Train the simple copy task.


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(src)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))
'''

# execute_example(example_simple_model)


@dataclass
class TrainState:
    step: int = 0
    accum_step: int = 0
    tokens: int = 0
    samples: int = 0

class LabelSmoothing(nn.Module):
    def __init__(self, size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.size = size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.crit = nn.KLDivLoss(reduction='sum')

    def forward(self, x: Tensor, tgt: Tensor):
        assert x.size(-1) == self.size
        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 2 is true label and padding idx
        true_dist.scatter_(1, tgt.unsqueeze(1), x.detach().clone().fill_(1-self.smoothing))
        true_dist[:, self.pad_idx] = 0
        pad_tgt = torch.nonzero(tgt == self.pad_idx)
        true_dist.index_fill_(0, pad_tgt.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.crit(x, true_dist.detach().clone())

class SimpleLossCompute(nn.Module):
    def __init__(self, generator, crit):
        super().__init__()
        self.crit = crit
        self.generator = generator

    def forward(self, x, tgt_y, norm):
        out = self.generator(x)
        loss = self.crit(
            out.reshape(-1, out.size(-1)), tgt_y.reshape(-1)
        )
        return loss, loss / norm
    
class Batch:
    def __init__(self, src: Tensor, tgt: Tensor, padding_idx=2):
        self.src = src
        self.src_mask = (src != padding_idx).unsqueeze(-2)
        self.tgt = tgt[:, :-1].detach().clone()
        self.tgt_y = tgt[:, 1:].detach().clone()
        self.tgt_mask = subsequent_mask(self.tgt.size(-1)) & (self.tgt != padding_idx).unsqueeze(-2)
        self.ntokens = (self.tgt_y != padding_idx).detach().sum()

def run_epoch(
        data_iter: list[Batch],
        model,
        loss_compute,
        optimizer: torch.optim.Adam,
        lr_scheduler: LambdaLR,
        mode='train',
        accum_iter=1,
        train_state=TrainState()
):
    start_time = time.time()
    n_accum = 0
    tokens = 0
    total_loss = 0
    total_tokens = 0
    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        tokens += batch.ntokens
        total_tokens += batch.ntokens
        train_state.step += 1
        train_state.tokens += batch.ntokens
        train_state.samples += batch.src.size(0)
        if 'train' in mode:
            loss_node.backward()
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            lr_scheduler.step()

        if 'train' in mode and i % 40 == 0:
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start_time = time.time()
            tokens = 0
        del loss_node, loss
    return total_loss / total_tokens, train_state

def data_gen(vocab, batch_size, batches):
    for _ in range(batches):
        src = torch.randint(1, vocab, (batch_size, 10)).type(torch.long)
        src[:, 0] = 1
        src = src.clone().detach()
        tgt = src.clone().detach()
        yield Batch(src, tgt, 0)

def greedy_decode(model: EncoderDecoder, src, src_mask, max_len, start_symbol):
    ys = torch.zeros(1, 1).fill_(start_symbol).type(torch.long)
    memory = model.encode(src, src_mask)
    for _ in range(max_len-1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(-1)))
        prob = model.generator(out[:, -1])
        _, next_ = torch.max(prob, dim=-1)
        ys = torch.concat((ys, next_.unsqueeze(0)), -1)
    return ys

def rate(step, warmup=4000, model_size=512, factor = 1.0):
    if step == 0:
        step = 1
    return factor * model_size ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)

def train_examples():
    V = 11
    batches = 20
    batch_size = 80
    lr = 0.5
    d_model = 512
    warm_up = 400
    model = make_model(V, V, N=2, d_model=d_model)
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: rate(x, warmup=warm_up, model_size=d_model, factor=1.0))
    criterion = LabelSmoothing(V, 0)
    loss_compute = SimpleLossCompute(model.generator, criterion)
    for _ in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, batches),
            model,
            loss_compute,
            optimizer,
            lr_scheduler,
            mode='train',
            accum_iter=1,
            train_state=TrainState()
        )

        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            loss_compute,
            optimizer,
            lr_scheduler,
            mode='eval',
            accum_iter=1,
            train_state=TrainState()
        )
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    print(src)
    src_mask = torch.ones(1, 1, 10)
    max_len = src.shape[1]
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

for _ in range(10):
    train_examples()