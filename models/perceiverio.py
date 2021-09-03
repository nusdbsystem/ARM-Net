# lucidrains/perceiver-pytorch
from ray import tune
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from models.layers import Embedding

# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# main class

class PerceiverIO(nn.Module):
    def __init__(self, *, depth, in_dim, n_in_query=512, in_query_dim=512, n_cross_head=1, n_self_head=8,
                 cross_dim=64, self_dim=64, weight_sharing=False, self_per_cross=1, out_query_dim, logits_dim=None):
        '''
        :param depth: Number of blocks of (cross_attn, self_per_cross_attn*self_attn)
        :param in_dim: Number of dimensions per input feature (input embedding size)
        :param n_in_query: Number of input queries
        :param in_query_dim: Number of dimensions per latent query (latent query embedding size)
        :param n_cross_head: Number of heads for each cross attention
        :param n_self_head: Number of heads for each latent self attention
        :param cross_dim: Number of dimensions per cross attention head
        :param self_dim: Number of dimensions per latent self attention head
        :param weight_sharing: Whether to tie self_attn layers within each block
        :param self_per_cross: Number of self_attn per cross_attn
        :param out_query_dim: Number of dimensions per output query (output query embedding size)
        :param logits_dim: Number of dimensions per output (output embedding size)
        '''
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_in_query, in_query_dim))

        get_cross_attn = lambda: PreNorm(in_query_dim,
                                         Attention(in_query_dim, in_dim, heads=n_cross_head, dim_head=cross_dim),
                                         context_dim=in_dim)
        get_cross_ff = lambda: PreNorm(in_query_dim, FeedForward(in_query_dim))
        get_latent_attn = lambda: PreNorm(in_query_dim,
                                          Attention(in_query_dim, heads=n_self_head, dim_head=self_dim))
        get_latent_ff = lambda: PreNorm(in_query_dim, FeedForward(in_query_dim))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_sharing
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.decoder_cross_attn = PreNorm(out_query_dim, Attention(out_query_dim, in_query_dim, heads=n_cross_head,
                                                                   dim_head=cross_dim), context_dim=in_query_dim)
        self.to_logits = nn.Linear(out_query_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(self, data, mask=None, out_query=None):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b=b)
        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        if not exists(out_query):
            return x

        # cross attend from decoder queries to latents
        latents = self.decoder_cross_attn(out_query, context=x)

        # final linear out
        return self.to_logits(latents)


PERCEIVER_CONFIG = {
    # training config
    'lr': 1e-3,
    # model config - grid search
    'nemb': tune.grid_search([2, 4, 8, 16, 32]),
    'depth': tune.grid_search([1, 2, 4]),
    'n_in_query': tune.grid_search([4, 8, 16, 32]),
    'n_attn_head': 8,
    'hid_dim': tune.grid_search([4, 8, 16, 32]),
}

class PerceiverTab(nn.Module):
    """
    Model: Perceiver IO for Structured Data
    Ref:   A. Jaegle, et al. Perceiver IO: A General Architecture for Structured Inputs & Outputs, 2021.
    """
    def __init__(self, nclass, nfield, nfeat, nemb, depth, n_in_query=128, n_attn_head=8, hid_dim=64):
        super().__init__()
        # input feature embedding & position encoding
        self.feat_emb = Embedding(nfeat, nemb)
        self.pos_emb = nn.Embedding(nfield, nemb)

        # output query per class
        self.out_query = nn.Parameter(torch.randn(nclass, hid_dim))

        self.perceiver_io = PerceiverIO(depth=depth, in_dim=nemb, n_in_query=n_in_query, in_query_dim=hid_dim,
                                        n_cross_head=1, n_self_head=n_attn_head, cross_dim=hid_dim, self_dim=hid_dim,
                                        weight_sharing=False, self_per_cross=1, out_query_dim=hid_dim, logits_dim=1)

    def forward(self, x):
        """
        :param x:       FloatTensor B*F
        :return:        y of size B, Regression and Classification (+softmax)
        """
        bsz, nfield, device = x.shape[0], x.shape[1], x.device
        x = self.feat_emb(x)                                                    # B*F*E

        pos_emb = self.pos_emb(torch.arange(nfield, device=device))
        pos_emb = rearrange(pos_emb, 'f e -> () f e')                           # 1*F*E
        x = x + pos_emb                                                         # B*F*E

        out_query = repeat(self.out_query, 'nclass d -> b nclass d', b=bsz)
        logits = self.perceiver_io(x, mask=None, out_query=out_query)           # B*nclass*1
        return logits.squeeze(2)                                                # B*nclass