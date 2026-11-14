import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossKeypointLearning(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Global cross-space attention
        self.conv_h = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.conv_k = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Local joint relations
        self.conv_local = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)

        # Global descriptor integration
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q_kd):
        """
        Args:
            q_kd: (B, N, K, C)
        Returns:
            refined: (B, N, K, C)
        """
        B, N, K, C = q_kd.shape
        assert C == self.d_model

        # Reshape to (B, C, N, K) for spatial convolutions
        x = q_kd.permute(0, 3, 1, 2)  # (B, C, N, K)

        # --- Global cross-space attention ---
        # Pool along keypoint dimension (K -> 1), keep person dimension N
        q_h = F.adaptive_avg_pool2d(x, (N, 1))  # (B, C, N, 1)
        a_h = self.sigmoid(self.conv_h(q_h))    # (B, C, N, 1)

        # Pool along person dimension (N -> 1), keep keypoint dimension K
        q_k = F.adaptive_avg_pool2d(x, (1, K))  # (B, C, 1, K)
        a_k = self.sigmoid(self.conv_k(q_k))    # (B, C, 1, K)

        # Broadcast and apply attention to original features
        q_global = x * a_h * a_k  # (B, C, N, K)

        # --- Local joint relations ---
        q_local = self.conv_local(x)  # (B, C, N, K)

        # --- Feature integration ---
        # Cross-weight the two branches with complementary channel descriptors.
        q_wg = self.softmax(F.adaptive_avg_pool2d(q_global, (1, 1)))
        q_wl = self.softmax(F.adaptive_avg_pool2d(q_local, (1, 1)))
        local_attention = (q_local * q_wg).sum(dim=1, keepdim=True)
        global_attention = (q_global * q_wl).sum(dim=1, keepdim=True)
        out = x * (local_attention + global_attention)

        # Reshape back to (B, N, K, C)
        out = out.permute(0, 2, 3, 1)
        return out


class KeypointSelfAttention(nn.Module):
    """Keypoint self-attention for contextual reasoning over body joints."""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B*N, K, C)
        Returns:
            out: (B*N, K, C)
        """
        attn_out, _ = self.self_attn(x, x, x)
        return self.norm(x + attn_out)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 4
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.fc2(self.dropout1(self.activation(self.fc1(x)))))


class TopologyAwareKeypointRefinement(nn.Module):

    def __init__(self, d_model, num_keypoints=None, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_keypoints = num_keypoints

        self.cross_keypoint_learning = CrossKeypointLearning(d_model)
        self.self_attention = KeypointSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, N, K, C) where B=batch, N=num person queries, K=num keypoints
        Returns:
            refined: (B, N, K, C)
        """
        B, N, K, C = x.shape

        # 1) Cross-keypoint learning (global + local topology)
        x = self.cross_keypoint_learning(x)

        # 2) Keypoint self-attention
        x_flat = x.reshape(B * N, K, C)  # (B*N, K, C)
        x_flat = self.self_attention(x_flat)

        # 3) Feed-forward network
        x_flat = self.norm2(x_flat + self.ffn(x_flat))

        # Reshape back
        x = x_flat.view(B, N, K, C)
        return x
