import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as torch_init
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    elif classname.find('GRU') != -1: pass

# --- (CIBE) ---
class CausalInfoBottleneckEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=None, latent_dim=None):
        super().__init__()
        if hidden_dim is None: hidden_dim = feature_dim
        if latent_dim is None: latent_dim = feature_dim
        self.encoder_gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_mu.apply(weights_init)
        self.fc_logvar.apply(weights_init)
    def forward(self, visual_feat):
        h, _ = self.encoder_gru(visual_feat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

#---B(LPM using Conv1d) ---
class PatternMonitoringPathway(nn.Module):
    def __init__(self, feature_dim, conv_hidden_dim=None, kernel_size=5, dropout_rate=0.1):
        super().__init__()
        if conv_hidden_dim is None:
            conv_hidden_dim = feature_dim
        padding = kernel_size // 2
        self.pmp_conv = nn.Sequential(
            nn.Conv1d(feature_dim, conv_hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(conv_hidden_dim, feature_dim, kernel_size=1)
        )
        self.norm_out = nn.LayerNorm(feature_dim)
        self.pmp_conv.apply(weights_init)
    def forward(self, visual_feat):
        x_permuted = visual_feat.permute(0, 2, 1)
        pmp_features_permuted = self.pmp_conv(x_permuted)
        pmp_features = pmp_features_permuted.permute(0, 2, 1)
        pmp_features = self.norm_out(pmp_features)
        return pmp_features

# ---(CGF) ---
class ContextualGatingFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.linear_gate = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.linear_gate.apply(weights_init)

    def forward(self, main_features, context_features):
        """
        Args:
            main_features (torch.Tensor): Main stream features (mu) [B, T, D]
            context_features (torch.Tensor): Context stream features (pmp patterns) [B, T, D]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - Fused/modulated features [B, T, D]
              - Gate tensor [B, T, D]
        """
        gate = torch.sigmoid(self.linear_gate(context_features)) # [B, T, D]
        processed_features = main_features * gate
        processed_features = self.norm(processed_features)
        # ***** 修改点 1: 返回 gate *****
        return processed_features, gate

# --- (TIBL) ---
# (保持不变)
class TemporalInfoBottleneckLoss(nn.Module):
    def __init__(self, latent_dim, feature_dim, kl_beta=0.01, loss_weight=0.1, normal_thresh=0.5):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, (latent_dim + feature_dim) // 2), nn.ReLU(),
            nn.Linear((latent_dim + feature_dim) // 2, feature_dim)
        )
        self.pred_loss_fn = nn.MSELoss(reduction='none')
        self.kl_beta = kl_beta
        self.loss_weight = loss_weight
        self.normal_thresh = normal_thresh
        self.predictor.apply(weights_init)
    def forward(self, mu, logvar, original_visual, element_scores, seq_len):
        B, T, D_latent = mu.shape
        device = mu.device
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        if T <= 1: pred_loss_padded = torch.zeros_like(kl_div)
        else:
            mu_prev = mu[:, :-1, :]
            x_target = original_visual[:, 1:, :]
            x_pred = self.predictor(mu_prev)
            pred_loss_t_minus_1 = self.pred_loss_fn(x_pred, x_target).mean(dim=-1)
            pred_loss_padded = F.pad(pred_loss_t_minus_1, (1, 0), "constant", 0)
        arange_time = torch.arange(T, device=device).expand(B, T)
        mask_time = arange_time < seq_len.unsqueeze(1)
        mask_normal = (element_scores.squeeze(-1) < self.normal_thresh) & mask_time
        if mask_normal.any(): masked_kl_loss = kl_div[mask_normal].mean()
        else: masked_kl_loss = torch.tensor(0.0, device=device, requires_grad=True)
        mask_normal_pred = mask_normal & (arange_time > 0)
        if mask_normal_pred.any(): masked_pred_loss = pred_loss_padded[mask_normal_pred].mean()
        else: masked_pred_loss = torch.tensor(0.0, device=device, requires_grad=True)
        aux_loss = masked_pred_loss + self.kl_beta * masked_kl_loss
        return aux_loss * self.loss_weight

class DS_IB(nn.Module):
    def __init__(self, feature_size, sample_size=None, num_classes=None, nc=None, nem=None, N=None, M=None, K=None): #
        super().__init__()
        self.feature_size = feature_size
        self.cibe_encoder = CausalInfoBottleneckEncoder(feature_dim=feature_size, latent_dim=feature_size)
        latent_dim = feature_size
        self.pmp = PatternMonitoringPathway(feature_dim=feature_size)
        self.fusion_gate = ContextualGatingFusion(feature_dim=latent_dim)
        processed_feature_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(processed_feature_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.aux_loss_module = TemporalInfoBottleneckLoss(latent_dim=latent_dim, feature_dim=feature_size)
        self.apply(weights_init)

    def forward(self, x, text_features=None, seq_len=None, is_training=True):
        target_device = x.device
        B, T, D = x.shape
        mu, logvar = self.cibe_encoder(x)
        pmp_features = self.pmp(x)


        processed_features, gate = self.fusion_gate(mu, pmp_features)
        # print(gate.shape)
        # print(gate)

        raw_logits = self.classifier(processed_features)
        # print(raw_logits.shape)
        element_scores = self.sigmoid(raw_logits)

        if is_training:
            seq_len = torch.tensor(seq_len, dtype=torch.long, device=target_device)
            aux_loss = self.aux_loss_module(mu, logvar, x, element_scores, seq_len)
            return raw_logits, element_scores, processed_features, aux_loss, gate

        else:

            return raw_logits, element_scores, processed_features, gate