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
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        return self.encoder(src, src_mask)
    
    def decode(self, memory, tgt, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        return self.decoder(memory, tgt, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)

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
    def __init__(self, attn, ffn, size, dropout):
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
    def __init__(self, src_attn, self_attn, ffn, size, dropout):
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

class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab, d_model)

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pos_en = torch.zeros((max_len, d_model))
        div_term = torch.exp(-math.log(10000) * torch.arange(0, d_model, 2) / d_model)
        pos_en[:, 0::2] = torch.sin(torch.arange(0, max_len).unsqueeze(1) * div_term)
        pos_en[:, 1::2] = torch.cos(torch.arange(0, max_len).unsqueeze(1) * div_term)
        self.register_buffer('pos_en', pos_en)

    def forward(self, x):
        return x + self.pos_en[x.size(-2)]
    
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.linear1(x).relu()))

def attention(query, key, value, mask=None, dropout=None):
    "Implementation of attention layer."
    d_model = query.size(-1)
    score = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(d_model)
    if mask is not None:
        score = score.masked_fill(mask, -1e9)
    p_attn = score.softmax(dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    out = torch.matmul(p_attn, value)
    return out, p_attn

def subsequent_mask(n):
    mask = torch.triu(torch.ones(1, n, n), diagonal=1)
    return mask == 0

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=None):
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
        dropout=0.1,
        d_model=512,
        d_ffn=2048,
        n_head=8
) -> EncoderDecoder :
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_head, d_model, dropout)
    ffn = FeedForwardLayer(d_model, d_ffn, dropout)
    src_embed = nn.Sequential(Embedding(d_model, src_vocab), PositionEncoding(d_model))
    tgt_embed = nn.Sequential(Embedding(d_model, tgt_vocab), PositionEncoding(d_model))
    generator = Generator(d_model, tgt_vocab)
    model = EncoderDecoder(
        c(src_embed),
        c(tgt_embed),
        Encoder(EncoderLayer(c(attn), c(ffn), d_model, dropout), N),
        Decoder(DecoderLayer(c(attn), c(attn), c(ffn), d_model, dropout), N),
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

# ========================= Training =========================
# ============================================================
# ============================================================

from dataclasses import dataclass
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
import time

@dataclass
class TrainState:
    step: int = 0
    accum_ste: int = 0
    tokens: int = 0
    samples: int = 0

class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.1, pad_idx=0):
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
    
class SimpleLoss(nn.Module):
    def __init__(self, crit, generator):
        super().__init__()
        self.crit = crit
        self.generator = generator

    def forward(self, x, tgt_y, norm):
        out = self.generator(x)
        loss = self.crit(
            out.view(-1, out.size(-1)), tgt_y.view(-1)
        )
        return loss / norm, loss
    
class Batch:
    def __init__(self, src: Tensor, tgt: Tensor, padding_idx=2):
        self.src = src
        self.src_mask = (src == padding_idx).unsqueeze(-2)
        self.tgt = tgt[:, :-1].detach().clone()
        self.tgt_y = tgt[:, 1:].detach().clone()
        self.tgt_mask = subsequent_mask(self.tgt.size(-1)) & (self.tgt != padding_idx).unsqueeze(-2)
        self.ntokens = (self.tgt_y != padding_idx).detach().sum()

def run_epoch(
        optimizer: torch.optim.Adam,
        lr_scheduler: LambdaLR,
        data_iter: list[Batch],
        model,
        loss_compute,
        accum_steps,
        mode='train',
        train_state=TrainState()
):
    start_time = time.time()
    n_accum = 0
    tokens = 0
    total_loss = 0
    total_tokens = 0
    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss_node, loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        tokens += batch.ntokens
        total_tokens += batch.ntokens
        train_state.step += 1
        train_state.tokens += batch.ntokens
        train_state.samples += batch.src.size(0)
        if 'train' in mode:
            loss_node.backward()
            if i % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_ste += 1
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

def greedy_decode(model: EncoderDecoder, src, src_mask, max_len, start_index):
    ys = torch.zeros(1, 1).fill_(start_index).type(torch.long)
    memory = model.encode(src, src_mask)
    for _ in range(max_len-1):
        out = model.decode(memory, ys, src_mask, subsequent_mask(ys.size(-1)))
        prob = model.generator(out[:, -1])
        _, next_ = torch.max(prob, dim=-1)
        ys = torch.concat((ys, next_.unsqueeze(0)), -1)
    return ys

def rate(step, warmup=4000, d_model=512, factor = 1.0):
    if step == 0:
        step = 1
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)

def train_examples():
    V = 11
    batches = 20
    batch_size = 80
    lr = 0.5
    d_model = 512
    warm_up = 400
    model = make_model(V, V, N=2, d_model=d_model)
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: rate(x, warmup=warm_up, d_model=d_model))
    criterion = LabelSmoothing(V)
    loss_compute = SimpleLoss(criterion, model.generator)
    for _ in range(20):
        model.train()
        run_epoch(
            optimizer,
            lr_scheduler,
            data_gen(V, batch_size, batches),
            model,
            loss_compute,
            1,
            mode='train',
            train_state=TrainState()
        )

        model.eval()
        run_epoch(
            optimizer,
            lr_scheduler,
            data_gen(V, batch_size, 5),
            model,
            loss_compute,
            1,
            mode='eval',
            train_state=TrainState()
        )
    src = torch.tensor([range(1, 11)]).type(torch.long)
    print(src)
    src_mask = torch.ones(1, 1, 10)
    ys = greedy_decode(model, src, src_mask, 10, 1)
    print(ys)

train_examples()