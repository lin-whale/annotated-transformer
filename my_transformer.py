import copy
import spacy.tokenizer
import torch
import math
from torch import nn
from torch.nn.functional import log_softmax
import gc


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


############################################################################
############################################################################
############################################################################

# class EncoderDecoder(nn.Module):
#     """
#     A standard Encoder-Decoder architecture. Base for this and many
#     other models.
#     """

#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator

#     def forward(self, src, tgt, src_mask, tgt_mask):
#         "Take in and process masked src and target sequences."
#         return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

#     def encode(self, src, src_mask):
#         return self.encoder(self.src_embed(src), src_mask)

#     def decode(self, memory, src_mask, tgt, tgt_mask):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

# class Generator(nn.Module):
#     "Define standard linear + softmax generation step."

#     def __init__(self, d_model, vocab):
#         super(Generator, self).__init__()
#         self.proj = nn.Linear(d_model, vocab)

#     def forward(self, x):
#         return log_softmax(self.proj(x), dim=-1)
    

# def clones(module, N):
#     "Produce N identical layers."
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# class Encoder(nn.Module):
#     "Core encoder is a stack of N layers"

#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)

#     def forward(self, x, mask):
#         "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)
    

# class LayerNorm(nn.Module):
#     "Construct a layernorm module (See citation for details)."

#     def __init__(self, features, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.a_2 = nn.Parameter(torch.ones(features))
#         self.b_2 = nn.Parameter(torch.zeros(features))
#         self.eps = eps

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

# class SublayerConnection(nn.Module):
#     """
#     A residual connection followed by a layer norm.
#     Note for code simplicity the norm is first as opposed to last.
#     """

#     def __init__(self, size, dropout):
#         super(SublayerConnection, self).__init__()
#         self.norm = LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, sublayer):
#         "Apply residual connection to any sublayer with the same size."
#         return x + self.dropout(sublayer(self.norm(x)))
    

# class EncoderLayer(nn.Module):
#     "Encoder is made up of self-attn and feed forward (defined below)"

#     def __init__(self, size, self_attn, feed_forward, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 2)
#         self.size = size

#     def forward(self, x, mask):
#         "Follow Figure 1 (left) for connections."
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#         return self.sublayer[1](x, self.feed_forward)
    

# class Decoder(nn.Module):
#     "Generic N layer decoder with masking."

#     def __init__(self, layer, N):
#         super(Decoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)

#     def forward(self, x, memory, src_mask, tgt_mask):
#         for layer in self.layers:
#             x = layer(x, memory, src_mask, tgt_mask)
#         return self.norm(x)
    

# class DecoderLayer(nn.Module):
#     "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

#     def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
#         super(DecoderLayer, self).__init__()
#         self.size = size
#         self.self_attn = self_attn
#         self.src_attn = src_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 3)

#     def forward(self, x, memory, src_mask, tgt_mask):
#         "Follow Figure 1 (right) for connections."
#         m = memory
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
#         x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
#         return self.sublayer[2](x, self.feed_forward)
    

# def subsequent_mask(size):
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
#         torch.uint8
#     )
#     return subsequent_mask == 0


# def attention(query, key, value, mask=None, dropout=None):
#     "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = scores.softmax(dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn


# class MultiHeadedAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = clones(nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, query, key, value, mask=None):
#         "Implements Figure 2"
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         nbatches = query.size(0)

#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = [
#             lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#             for lin, x in zip(self.linears, (query, key, value))
#         ]

#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = attention(
#             query, key, value, mask=mask, dropout=self.dropout
#         )

#         # 3) "Concat" using a view and apply a final linear.
#         x = (
#             x.transpose(1, 2)
#             .contiguous()
#             .view(nbatches, -1, self.h * self.d_k)
#         )
#         del query
#         del key
#         del value
#         return self.linears[-1](x)
    

# class PositionwiseFeedForward(nn.Module):
#     "Implements FFN equation."

#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = nn.Linear(d_model, d_ff)
#         self.w_2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.w_2(self.dropout(self.w_1(x).relu()))
    

# class Embeddings(nn.Module):
#     def __init__(self, d_model, vocab):
#         super(Embeddings, self).__init__()
#         self.lut = nn.Embedding(vocab, d_model)
#         self.d_model = d_model

#     def forward(self, x):
#         return self.lut(x) * math.sqrt(self.d_model)
    

# class PositionalEncoding(nn.Module):
#     "Implement the PE function."

#     def __init__(self, d_model, dropout, max_len=20000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         # position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000) / d_model)
#         pe[:, 0::2] = torch.sin(torch.arange(0, max_len).unsqueeze(1) * div_term)
#         pe[:, 1::2] = torch.cos(torch.arange(0, max_len).unsqueeze(1) * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         x = x + self.pe[:, : x.size(1)].requires_grad_(False)
#         return self.dropout(x)
    

# def make_model(
#     src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
# ):
#     "Helper: Construct a model from hyperparameters."
#     c = copy.deepcopy
#     attn = MultiHeadedAttention(h, d_model)
#     ff = PositionwiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)
#     model = EncoderDecoder(
#         Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
#         Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
#         nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
#         nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
#         Generator(d_model, tgt_vocab),
#     )

#     # This was important from their code.
#     # Initialize parameters with Glorot / fan_avg.
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model


# ========================= Training =========================
# ============================================================
# ============================================================

from dataclasses import dataclass
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
import time


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


# def run_epoch(
#     data_iter,
#     model,
#     loss_compute,
#     optimizer,
#     scheduler,
#     mode="train",
#     accum_iter=1,
#     train_state=TrainState(),
# ):
#     """Train a single epoch"""
#     start = time.time()
#     total_tokens = 0
#     total_loss = 0
#     tokens = 0
#     n_accum = 0
#     for i, batch in enumerate(data_iter):
#         out = model.forward(
#             batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
#         )
#         loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
#         # loss_node = loss_node / accum_iter
#         if mode == "train" or mode == "train+log":
#             loss_node.backward()
#             train_state.step += 1
#             train_state.samples += batch.src.shape[0]
#             train_state.tokens += batch.ntokens
#             if i % accum_iter == 0:
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)
#                 n_accum += 1
#                 train_state.accum_step += 1
#             scheduler.step()

#         total_loss += loss
#         total_tokens += batch.ntokens
#         tokens += batch.ntokens
#         if i % 40 == 0 and (mode == "train" or mode == "train+log"):
#             lr = optimizer.param_groups[0]["lr"]
#             elapsed = time.time() - start
#             print(
#                 (
#                     "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
#                     + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
#                 )
#                 % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
#             )
#             start = time.time()
#             tokens = 0
#         del loss
#         del loss_node
#     return total_loss / total_tokens, train_state


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


# execute_example(example_simple_model)


@dataclass
class TrainState:
    step: int = 0
    accum_step: int = 0
    tokens: int = 0
    samples: int = 0

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.size = size
        self.pad_idx = padding_idx
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

class SimpleLossCompute:
    def __init__(self, generator, crit):
        super().__init__()
        self.crit = crit
        self.generator = generator

    def __call__(self, x, tgt_y, norm):
        out = self.generator(x)
        loss = self.crit(
            out.reshape(-1, out.size(-1)), tgt_y.reshape(-1)
        )
        return loss.data, loss / norm
    
class Batch:
    def __init__(self, src: Tensor, tgt: Tensor, padding_idx=2):
        self.src = src
        self.src_mask = (src != padding_idx).unsqueeze(-2)
        self.tgt = tgt[:, :-1].detach().clone()
        self.tgt_y = tgt[:, 1:].detach().clone()
        self.tgt_mask = subsequent_mask(self.tgt.size(-1)).type_as(self.tgt.detach()) & (self.tgt != padding_idx).unsqueeze(-2)
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
    return total_loss, train_state

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

# for _ in range(10):
#     train_examples()

from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
import spacy
import os
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, pad

def load_tokenizer():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_news_sm")
        spacy_en = spacy.load("en_core_web_sm")
    return spacy_de, spacy_en

from torchtext.datasets import multi30k
from torchtext import datasets
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

multi30k.MD5["train"] = "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e"
multi30k.MD5["valid"] = "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c"
multi30k.MD5["test"] = "6d1ca1dba99e2c5dd54cae1226ff11c2551e6ce63527ebb072a1f70f72a5cd36"

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def build_vocab(spacy_de, spacy_en):
    def tokenizer_de(text):
        return tokenize(text, spacy_de)
    
    def tokenizer_en(text):
        return tokenize(text, spacy_en)

    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_de = build_vocab_from_iterator(
        yield_tokens(train+val+test, tokenizer_de, 0),
        min_freq=2,
        specials=['<s>', '</s>', '<blank>', '<unknown>']
    )

    vocab_en = build_vocab_from_iterator(
        yield_tokens(train+val+test, tokenizer_en, 1),
        min_freq=2,
        specials=['<s>', '</s>', '<blank>', '<unknown>']
    )

    vocab_de.set_default_index(vocab_de['<blank>'])
    vocab_en.set_default_index(vocab_en['<blank>'])
    return vocab_de, vocab_en

def load_vocab(spacy_de, spacy_en):
    if not os.path.exists('vocab.pt'):
        vocab_de, vocab_en = build_vocab(spacy_de, spacy_en)
        torch.save((vocab_de, vocab_en), 'vocab.pt')
    else:
        vocab_de, vocab_en = torch.load('vocab.pt')
    return vocab_de, vocab_en

spacy_de, spacy_en = load_tokenizer()
vocab_de, vocab_en = load_vocab(spacy_de, spacy_en)

def collate_batch(
        max_padding,
        pad_idx,
        batch,
        vocab_de,
        vocab_en,
        pipeline_de,
        pipeline_en,
        device
):
    src_list = []
    tgt_list = []
    st, eos = vocab_de['<s>'], vocab_de['</s>']
    st_id = torch.tensor([st], dtype=torch.long, device=device)
    eos_id = torch.tensor([eos], dtype=torch.long, device=device)
    for src, tgt in batch:
        src_ = torch.concat(
            [
                st_id.clone(),
                torch.tensor(vocab_de(pipeline_de(src)), dtype=torch.long, device=device),
                eos_id.clone()
            ],
            dim=0
        )
        tgt_ = torch.concat(
            [
                st_id.clone(),
                torch.tensor(vocab_en(pipeline_en(tgt)), dtype=torch.long, device=device),
                eos_id.clone()
            ],
            dim=0
        )

        src_ = pad(src_, [0, max_padding-len(src_)], value=pad_idx)
        tgt_ = pad(tgt_, [0, max_padding-len(tgt_)], value=pad_idx)
        src_list.append(src_)
        tgt_list.append(tgt_)
    return (
        torch.stack(src_list, dim=0),
        torch.stack(tgt_list, dim=0)
    )

def build_dataloaders(
    device,
    spacy_de,
    spacy_en,
    vocab_de,
    vocab_en,
    config,
):

    def tokenizer_de(text):
        return tokenize(text, spacy_de)
    
    def tokenizer_en(text):
        return tokenize(text, spacy_en)

    pad_idx = vocab_de['<blank>']
    def collate_fn(batch):
        return collate_batch(
            max_padding=config['max_padding'],
            pad_idx=pad_idx,
            batch=batch,
            vocab_de=vocab_de,
            vocab_en=vocab_en,
            pipeline_de=tokenizer_de,
            pipeline_en=tokenizer_en,
            device=device
        )

    train, val, test = datasets.Multi30k(language_pair=('de', 'en'))
    train_map = to_map_style_dataset(train)
    val_map = to_map_style_dataset(val)
    train_dataloader = DataLoader(
        train_map,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_map,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader

# config = {
#     'max_padding': 128,
#     'batch_size': 80
# }
# train_data, _ = build_dataloaders(0, spacy_de, spacy_en, vocab_de, vocab_en, config)
# for i in train_data:
#     print(i)


def train_worker(
        gpu,
        spacy_de,
        spacy_en,
        vocab_de,
        vocab_en,
        config
):
    train_dataloader, val_dataloader = build_dataloaders(
        gpu,
        spacy_de,
        spacy_en,
        vocab_de,
        vocab_en,
        config
    )
    model = make_model(len(vocab_de), len(vocab_en), N=6)
    d_model = model.encoder.layers[0].size
    model.cuda(gpu)

    pad_idx = vocab_en['<blank>']
    criterion = LabelSmoothing(len(vocab_en), pad_idx, 0.1)
    criterion.cuda(gpu)

    loss_compute = SimpleLossCompute(model.generator, criterion)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['base_lr'],
        betas=(0.9, 0.98),
        eps=1e-9
    )

    lr_scheduler = LambdaLR(optimizer, lambda x: rate(x, model_size=d_model, factor=1.0, warmup=config['warmup']))
    for i in range(config['num_epochs']):
        model.train()
        train_state = TrainState()
        sloss, train_state = run_epoch(
            (Batch(src, tgt) for (src, tgt) in train_dataloader),
            model,
            loss_compute,
            optimizer,
            lr_scheduler,
            'train',
            config['accum_iter'],
            train_state
        )

        torch.save(model.state_dict(),
                   "{}{:2d}.pt".format(config['file_prefix'], i))
        torch.cuda.empty_cache()

        model.eval()
        run_epoch(
            (Batch(src, tgt) for (src, tgt) in val_dataloader),
            model,
            loss_compute,
            optimizer,
            lr_scheduler,
            'eval'
        )
        torch.cuda.empty_cache()
    torch.save(
        model.state_dict(),
        "{}final.pt".format(config['file_prefix'])
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    train_worker(0, spacy_de, spacy_en, vocab_src, vocab_tgt, config)

def load_trained_model():
    config = {
        'batch_size': 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = config['file_prefix'] + 'final.pt'
    if not os.path.exists(model_path):
        train_model(vocab_de, vocab_en, spacy_de, spacy_en, config)
    
    model = make_model(len(vocab_de), len(vocab_en), N=6)
    model.load_state_dict(torch.load(model_path))
    return model

model = load_trained_model()


###########################################################################
###########################################################################
###########################################################################
###########################################################################

'''
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def is_interactive_notebook():
    return __name__ == "__main__"

def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


# %% id="t4BszXXJTsqL" tags=[]
def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


# %% id="jU3kVlV5okC-" tags=[]


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


if is_interactive_notebook():
    # global variables used later in the script
    spacy_de, spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])


# %% [markdown] id="-l-TFwzfTsqL"
#
# > Batching matters a ton for speed. We want to have very evenly
# > divided batches, with absolutely minimal padding. To do this we
# > have to hack a bit around the default torchtext batching. This
# > code patches their default batching to make sure we search over
# > enough sentences to find tight batches.

# %% [markdown] id="kDEj-hCgokC-" tags=[] jp-MarkdownHeadingCollapsed=true
# ## Iterators

# %% id="wGsIHFgOokC_" tags=[]
def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


# %% id="ka2Ce_WIokC_" tags=[]
def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


# %% [markdown] id="90qM8RzCTsqM"
# ## Training the System

# %%
def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


# %% tags=[]
def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model



model = load_trained_model()
'''