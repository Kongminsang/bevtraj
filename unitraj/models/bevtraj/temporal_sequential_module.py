import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMHA(nn.Module):
    """
    Multi-head self attention without projection.

    q,k = concat(dec_embed, time_pe)
    v   = dec_embed

    Inputs
        dec_embed : [T, BK, D]
        time_pe   : [T, BK, D]

    Output
        out       : [T, BK, D]
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0

        self.D = d_model
        self.H = num_heads
        self.dh = d_model // num_heads
        self.scale = math.sqrt(2 * self.dh)

        self.dropout = dropout

    def forward(self, dec_embed, time_pe):

        T, BK, D = dec_embed.shape

        # [BK, H, T, dh]
        dec_h = (
            dec_embed.permute(1, 0, 2)
            .reshape(BK, T, self.H, self.dh)
            .permute(0, 2, 1, 3)
        )
        time_h = (
            time_pe.permute(1, 0, 2)
            .reshape(BK, T, self.H, self.dh)
            .permute(0, 2, 1, 3)
        )
        q = torch.cat([dec_h, time_h], dim=-1)  # [BK,H,T,2dh]
        k = torch.cat([dec_h, time_h], dim=-1)
        v = dec_h  # [BK,H,T,dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # [BK,H,T,dh]
        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(BK, T, D)
            .permute(1, 0, 2)
        )
        return out
    

class TemporalMHA_NoTimePE(nn.Module):
    """
    Multi-head self attention without projection and without time_pe.

    q,k,v = dec_embed

    Inputs
        dec_embed : [T, BK, D]
        time_pe   : unused (for drop-in compatibility)

    Output
        out       : [T, BK, D]
    """
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.D = d_model
        self.H = num_heads
        self.dh = d_model // num_heads
        self.scale = math.sqrt(self.dh)
        self.dropout = dropout

    def forward(self, dec_embed, time_pe=None):
        T, BK, D = dec_embed.shape

        # [BK, H, T, dh]
        dec_h = (
            dec_embed.permute(1, 0, 2)
            .reshape(BK, T, self.H, self.dh)
            .permute(0, 2, 1, 3)
        )

        q = dec_h
        k = dec_h
        v = dec_h

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # [BK,H,T,dh]
        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(BK, T, D)
            .permute(1, 0, 2)
        )
        return out