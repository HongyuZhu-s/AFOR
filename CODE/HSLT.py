"""
HSLT: Hierarchical Spatial Information Learning Model with Transformers
for EEG-Based Emotion Recognition.

Reference:
    Wang, Zhe et al., "Transformers for EEG-Based Emotion Recognition:
    A Hierarchical Spatial Information Learning Model," IEEE Sensors J., 2022.

Adapted from the LibEER PyTorch implementation.
Accepts (batch, channels, timepoints) input and outputs logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Building blocks
# ==============================================================================

class MLP(nn.Module):
    def __init__(self, hidden_states, output_states, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(output_states, hidden_states)
        self.fc2 = nn.Linear(hidden_states, output_states)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = F.gelu(self.fc1(x))
        hidden = self.dropout(hidden)
        output = F.gelu(self.fc2(hidden))
        return self.dropout(output)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads=16, msa_dimensions=64,
                 dropout_rate=0.4):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(model_dim, eps=1e-6)
        self.fc1 = nn.Linear(model_dim, msa_dimensions)
        self.attention = nn.MultiheadAttention(
            embed_dim=msa_dimensions, num_heads=num_heads,
            dropout=dropout_rate, batch_first=True)
        self.fc2 = nn.Linear(msa_dimensions, model_dim)
        self.layernorm2 = nn.LayerNorm(model_dim, eps=1e-6)
        self.mlp = MLP(model_dim * 4, model_dim)

    def forward(self, x):
        x1 = self.layernorm1(x)
        x1 = F.relu(self.fc1(x1))
        attn_out, _ = self.attention(x1, x1, x1)
        attn_out = F.relu(self.fc2(attn_out))
        x2 = x + attn_out
        x3 = self.layernorm2(x2)
        return x2 + self.mlp(x3)


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(model_dim) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class LinearEmbedding(nn.Module):
    def __init__(self, num_patches, in_channels, projection_dim,
                 expand=True, dropout=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.expand = expand

        self.class_token = nn.Parameter(torch.randn(1, projection_dim),
                                        requires_grad=True)
        self.projection = nn.Linear(in_channels, projection_dim)
        self.position_embedding = nn.Embedding(num_patches + 1,
                                               projection_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patch):
        batch = patch.size(0)

        if self.expand:  # electrode-level
            class_token = self.class_token.expand(batch, -1)
            class_token = class_token.view(batch, 1, self.projection_dim)
            patches_embed = self.projection(patch)
            patches_embed = torch.cat([class_token, patches_embed], 1)
            positions = torch.arange(0, self.num_patches + 1, 1,
                                     device=patch.device)
            positions_embed = self.position_embedding(positions)
            encoded = patches_embed + positions_embed
        else:  # brain-region-level
            class_token = self.class_token.expand(batch, -1)
            class_token = class_token.view(batch, 1, self.projection_dim)
            patches_embed = self.projection(patch)
            patches_embed = self.dropout(patches_embed)
            patches_embed = torch.cat([class_token, patches_embed], 1)
            positions = torch.arange(0, self.num_patches + 1, 1,
                                     device=patch.device)
            positions_embed = self.position_embedding(positions)
            encoded = patches_embed + positions_embed

        return encoded


# ==============================================================================
# Main HSLT model — adapted for (B, C, T) input
# ==============================================================================

class HSLT(nn.Module):
    """
    HSLT emotion recognition model.

    Args:
        num_electrodes: number of EEG channels (32 for DEAP, 62 for SEED)
        in_channels:    number of timepoints / feature dim per sample
        num_classes:    number of output classes
        De:             electrode-level embedding dim (default 8)
        Dr:             brain-region-level embedding dim (default 16)
        Le:             number of electrode-level transformer layers (default 2)
        Lr:             number of region-level transformer layers (default 2)
    """

    def __init__(self, num_electrodes=32, in_channels=128, num_classes=2,
                 De=8, Dr=16, Le=2, Lr=2):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.regions_num = 9

        # Brain region definitions per electrode count
        if num_electrodes == 6:   # ISRUC-Sleep (one region per channel)
            self.brain_regions = [
                {"name": "F3", "N": 1}, {"name": "C3", "N": 1},
                {"name": "O1", "N": 1}, {"name": "F4", "N": 1},
                {"name": "C4", "N": 1}, {"name": "O2", "N": 1},
            ]
            self.regions_num = 6
            self.regions_electrodes_num = 1
        elif num_electrodes == 22:  # BCI IV 2a
            self.brain_regions = [
                {"name": "Pre-Frontal",      "N": 2},   # Fz, FC3
                {"name": "Frontal-L",        "N": 2},   # FC1, FCz
                {"name": "Frontal-R",        "N": 2},   # FC2, FC4
                {"name": "Central-L",        "N": 3},   # C5, C3, C1
                {"name": "Central-M",        "N": 2},   # Cz, C2
                {"name": "Central-R",        "N": 3},   # C4, C6, CP3
                {"name": "Parietal-L",       "N": 3},   # CP1, CPz, CP2
                {"name": "Parietal-M",       "N": 2},   # CP4, P1
                {"name": "Parietal-R",       "N": 3},   # Pz, P2, POz
            ]
            self.regions_electrodes_num = 3
        elif num_electrodes == 32:  # DEAP
            self.brain_regions = [
                {"name": "Pre-Frontal",      "N": 4},
                {"name": "Frontal",          "N": 5},
                {"name": "Left Temporal",    "N": 3},
                {"name": "Central",          "N": 5},
                {"name": "Right Temporal",   "N": 3},
                {"name": "Left Parietal",    "N": 3},
                {"name": "Parietal",         "N": 3},
                {"name": "Right Parietal",   "N": 3},
                {"name": "Occipital",        "N": 3},
            ]
            self.regions_electrodes_num = 3
        elif num_electrodes == 62:  # SEED
            self.brain_regions = [
                {"name": "Pre-Frontal",      "N": 5},
                {"name": "Frontal",          "N": 9},
                {"name": "Left Temporal",    "N": 6},
                {"name": "Central",          "N": 7},
                {"name": "Right Temporal",   "N": 6},
                {"name": "Left Parietal",    "N": 8},
                {"name": "Parietal",         "N": 4},
                {"name": "Right Parietal",   "N": 7},
                {"name": "Occipital",        "N": 10},
            ]
            self.regions_electrodes_num = 4
        else:
            raise ValueError(f"Unsupported electrode count: {num_electrodes}")

        # Electrode-level transformers (one per brain region)
        self.transformers = nn.ModuleList()
        for region in self.brain_regions:
            N = region["N"]
            self.transformers.append(LinearEmbedding(N, in_channels, De))
            self.transformers.append(TransformerEncoder(De, Le))
            if N != self.regions_electrodes_num:
                self.transformers.append(
                    nn.Linear(N + 1, self.regions_electrodes_num + 1))
            else:
                self.transformers.append(None)  # placeholder

        # Brain-region-level transformer
        region_in_dim = (self.regions_electrodes_num + 1) * De
        self.regions_embeddings = LinearEmbedding(
            self.regions_num, region_in_dim, Dr, expand=False)
        self.regions_transformer = TransformerEncoder(Dr, Lr)

        # Classifier — outputs logits (no activation, for CrossEntropyLoss)
        self.classifier = nn.Linear(Dr, num_classes)

    def forward(self, inputs):
        # inputs: (B, num_electrodes, in_channels) = (B, C, T)
        transformer_outputs = []
        curr = 0

        for i in range(self.regions_num):
            patch_emb, patch_trans, projection = \
                self.transformers[i * 3:(i + 1) * 3]
            N = self.brain_regions[i]["N"]

            x = inputs[:, curr:curr + N, :]                    # (B, N, T)
            x = patch_emb(x)                                   # (B, N+1, De)
            x = patch_trans(x)                                 # (B, N+1, De)
            x = x.unsqueeze(1)                                 # (B, 1, N+1, De)

            if projection is not None:
                x = x.permute(0, 1, 3, 2)                     # (B, 1, De, N+1)
                x = projection(x)                              # (B, 1, De, 4)
                x = x.permute(0, 1, 3, 2)                     # (B, 1, 4, De)

            transformer_outputs.append(x)
            curr += N

        # Concatenate region outputs along region dimension
        x = torch.cat(transformer_outputs, dim=1)              # (B, 9, 4, De)

        # Brain-region-level transformer
        x = x.reshape(x.shape[0], x.shape[1],
                       x.shape[2] * x.shape[3])                # (B, 9, 4*De)
        x = self.regions_embeddings(x)                         # (B, 10, Dr)
        x = self.regions_transformer(x)                        # (B, 10, Dr)

        # Classify from class token (position 0)
        return self.classifier(x[:, 0, :])                     # (B, num_classes)
