import math

import numpy as np
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BEVTrajPreEncoder(nn.Module):
    def __init__(self, config):

        super(BEVTrajPreEncoder, self).__init__()

        self.config = config
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.k_attr = config['k_attr']
        self.D = config['d_model']
        self.L_enc = config['num_encoder_layers']
        self.dropout = config['dropout']
        self.num_heads = config['tx_num_heads']
        self.tx_hidden_size = config['tx_hidden_size']
        
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.D)))
        self.pos_encoder = PositionalEncoding(self.D, dropout=0.0, max_len=config['past_len'])

        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.D, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.D, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

    def process_observations(self, target, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [target, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # target stuff
        target_tensor = target[:, :, :self.k_attr]

        # agents stuff
        temp_masks = torch.cat((torch.ones_like(target[:, :, -1:]), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return target_tensor, opps_tensor, opps_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs, B, num_agents, _ = agents_emb.shape
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        temp_masks[:, -1][temp_masks.sum(-1) == T_obs] = False  # Ensure that agent's that don't exist don't make NaN.
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (num_agents), -1)),
                                src_key_padding_mask=temp_masks)
        return agents_temp_emb.view(T_obs, B, num_agents, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs, B, num_agents, _ = agents_emb.shape
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(num_agents, B * T_obs, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, num_agents))
        agents_soc_emb = agents_soc_emb.view(num_agents, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

    def forward(self, inputs):
        # 'True' in agents_mask indicates that the corresponding value is valid.
        agents_in, agents_mask = inputs['obj_trajs'], inputs['obj_trajs_mask']
        target_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'] \
                                            .view(-1, 1, 1, 1).repeat(1, 1, *agents_in.shape[-2:])).squeeze(1)
        target_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'] \
                                            .view(-1, 1, 1).repeat(1, 1, agents_mask.shape[-1])).squeeze(1)
        indices = [0, 1, 3, 4, 5, -4, -3, -7] # (x, y), (l, w, h), (vx, vy), (timestamp)      
        
        target_inform = target_in[:, :, indices]
        target_inform[..., -1] = target_inform[..., -1] - torch.max(target_inform[:, :, -1], dim=1, keepdim=True)[0]
        target_inform[..., -1] = target_inform[..., -1] * target_mask.float()                                                                                                        
        target_in = torch.cat([target_inform, target_mask.unsqueeze(-1)], dim=-1)
        
        agents_inform = agents_in[:, :, :, indices]
        agents_inform[..., -1] = agents_inform[..., -1] - torch.max(agents_inform[:, 0, :, -1], dim=1, keepdim=True)[0].unsqueeze(-1)
        agents_inform[..., -1] = agents_inform[..., -1] * agents_mask.float()
        agents_in = torch.cat([agents_inform, agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        
        pre_encoder_emb = self._forward(target_in, agents_in)

        return pre_encoder_emb

    def _forward(self, target_in, agents_in):
        '''
        :param target_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, num_agents, k_attr+1] with last values being the existence mask.
        :return:
            pre_encoder_emb: shape [B, num_agents, T_obs, D]
        '''
        target_tensor, _agents_tensor, opps_masks = self.process_observations(target_in, agents_in)
        agents_tensor = torch.cat((target_tensor.unsqueeze(2), _agents_tensor), dim=2)
        
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)
        for i in range(self.L_enc):
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])

        pre_encoder_emb = agents_emb[:, :, 1:, :].permute(1, 2, 0, 3)
        
        return pre_encoder_emb


