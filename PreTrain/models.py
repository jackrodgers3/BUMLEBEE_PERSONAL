import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.init import constant_
import matplotlib.pyplot as plt


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "leaky_relu":
        return F.leaky_relu

    raise RuntimeError("activation should be relu/gelu/leaky_relu, not {}".format(activation))

def init_method_normal(sigma):
    """Init method based on N(0, sigma^2)."""
    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_

class EncoderDecoder(nn.Module):
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
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
        
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])    
    
    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, elementwise_affine=True)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)    


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, encoder_var=1, ln_gain_mult=1):
        super(SublayerConnection, self).__init__()
        self.ln_gain_mult = ln_gain_mult
        self.norm = nn.LayerNorm(size, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.ln_gain_mult * self.norm(x)))   
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout, ln_gain_mult=1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, ln_gain_mult=ln_gain_mult), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, ln_gain_mult=1):
        super(Decoder, self).__init__()
        self.ln_gain_mult = ln_gain_mult
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, elementwise_affine=True)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.ln_gain_mult * self.norm(x)    
    
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, ln_gain_mult):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, ln_gain_mult=ln_gain_mult), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
    
def attention(query, key, value, mask=None, dropout=None, attention_mult=1.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # 8 here is because base dimension of head is 64 so multiply by sqrt(64) to keep
    # attention scores as 1/sqrt(width) in base case
    scores = attention_mult * torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, encoder_var=1.0, bias=True, attention_mult=1.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = int(d_model // h)
        self.h = int(h)
        self.linears = clones(nn.Linear(d_model, d_model, bias=bias), 4)
        self.attention_mult = attention_mult
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.init_method = init_method_normal((encoder_var / d_model)**0.5)
        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(len(self.linears)):
            self.init_method(self.linears[i].weight)
            constant_(self.linears[i].bias, 0.)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, attention_mult=self.attention_mult)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, act_fn = "gelu", encoder_var=1, bias=True):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.init_method = init_method_normal((encoder_var / d_model)**0.5)
        self.act_fn = _get_activation_fn(act_fn)
        self.reset_parameters()

    def reset_parameters(self):
        self.init_method(self.w_1.weight)
        self.init_method(self.w_2.weight)
        if self.w_1.bias is not None:
            constant_(self.w_1.bias, 0.)
        if self.w_2.bias is not None:
            constant_(self.w_2.bias, 0.)

    def forward(self, x):
        return self.w_2(self.dropout(self.act_fn(self.w_1(x))))

    
class ParticleEventTransformer(nn.Module):
    def __init__(self, particle_dimensionality, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, act_fn = "relu", bias=True,
                 encoder_var=1, attention_mult=1, output_mult=1, use_mup=True, mask_particle_embedding=True,
                 mask_all_but_pid_embedding=False):
        super(ParticleEventTransformer, self).__init__()
        c = copy.deepcopy
        self.d_model = d_model
        self.bias = bias
        self.act_fn = act_fn
        attn = MultiHeadedAttention(h, d_model, attention_mult=attention_mult, encoder_var=encoder_var)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout, act_fn, encoder_var=encoder_var)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        
        self.particle_embedding = nn.Linear(particle_dimensionality, d_model)
        self.particle_embedding_init_method = init_method_normal((encoder_var / particle_dimensionality)**0.5)
        self.gen_reco_embedding = nn.Embedding(2, d_model)
        self.pid_embedding = nn.Embedding(100, d_model)
        self.mask_embedding = nn.Embedding(2, d_model)
        
        # defining "well-measured" values
        self.e_mass = 0.00051099895069
        self.mu_mass = 0.1056583755
        
        # replace below with MuReadout for mup performance
        # see https://github.com/microsoft/mup
        self.reco_layer = nn.Linear(d_model, particle_dimensionality)
        
        self.mask_particle_embedding = mask_particle_embedding
        self.mask_all_but_pid_embedding = mask_all_but_pid_embedding

        self.init_weights()
    
    def init_weights(self):
        self.particle_embedding_init_method(self.particle_embedding.weight)
        if self.particle_embedding.bias is not None:
            constant_(self.particle_embedding.bias, 0.)
        if self.bias:
            self.reco_layer.bias.data.zero_()
        self.reco_layer.weight.data.zero_()
        
    def forward(self, x):
        ids = x[:, :, 0].int()
        segment_ids = x[:, :, 1].int()
        four_vectors = x[:, :, 2:6].float()
        zerod_mask = x[:, :, -1].int()

        particle_embedding = self.particle_embedding(four_vectors)
        gen_reco_embedding = self.gen_reco_embedding(segment_ids)
        pid_embedding = self.pid_embedding(ids)
        mask_embedding = self.mask_embedding(zerod_mask)

        zerod_mask = zerod_mask[:, :, None].repeat(1, 1, self.d_model)
        if self.mask_particle_embedding:
            particle_embedding = particle_embedding * zerod_mask
        if self.mask_all_but_pid_embedding:
            particle_embedding = particle_embedding * zerod_mask
            gen_reco_embedding = gen_reco_embedding * zerod_mask
            mask_embedding = mask_embedding * zerod_mask

        "Take in and process masked reco and gen sequences."
        embedding = particle_embedding \
                    + gen_reco_embedding \
                    + pid_embedding \
                    + mask_embedding
        encoding = self.encode(embedding, None)
        
        out = self.reco_layer(encoding)
        
        # making substitutions
        # llbar phi
        #out[:, 7:9, 2] = x[:, 1:3, 4]
        
        #llbar eta
        #out[:, 7:9, 1] = x[:, 1:3, 3]
        
        return out
        

    def encode(self, embedding, src_mask):
        return self.encoder(embedding, src_mask)
 

def make_model(particle_dimensionality=4, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1, act_fn="gelu", encoder_var=1,
               attention_mult=1, output_mult=1, use_mup=False, mask_particle_embedding=True,
               mask_all_but_pid_embedding=False):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    model = ParticleEventTransformer(particle_dimensionality, N=N, d_model=d_model, d_ff=d_ff, h=h,
        dropout=dropout, act_fn = act_fn, encoder_var=encoder_var, attention_mult=attention_mult, output_mult=output_mult, use_mup=use_mup,
        mask_particle_embedding=True, mask_all_but_pid_embedding=False)
    
    if torch.cuda.is_available():
        model.cuda()
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model