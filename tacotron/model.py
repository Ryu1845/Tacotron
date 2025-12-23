import importlib_resources
import math

import toml
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for self-attention layers."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: [B, T, D] tensor
        Returns:
            [B, T, D] tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention and feed-forward."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [B, T, D] tensor
            key_padding_mask: [B, T] boolean mask (True = masked positions)
        Returns:
            [B, T, D] tensor
        """
        # Generate causal mask if needed
        attn_mask = None
        if self.causal:
            T = x.size(1)
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )

        # Self-attention with residual
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))
        return x


class Tacotron(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.input_size = 2 * decoder["input_size"]
        self.n_mels = decoder["n_mels"]
        self.reduction_factor = decoder["reduction_factor"]
        self.n_kernels = decoder["attention"]["n_kernels"]

        self.encoder = Encoder(**encoder)
        self.decoder = Decoder(**decoder)

    @classmethod
    def from_pretrained(cls, url, map_location=None, cfg_path=None):
        """
        Loads the Torch serialized object at the given URL
        (uses torch.hub.load_state_dict_from_url).

        Parameters:
            url (string): URL of the weights to download
            map_location:  a function or a dict specifying how to remap
                storage locations (see torch.load).
            cfg_path (Path): path to config file.
                Defaults to tacotron/config.toml
        """
        cfg_ref = (
            importlib_resources.files("tacotron").joinpath("config.toml")
            if cfg_path is None
            else cfg_path
        )
        with cfg_ref.open() as file:
            cfg = toml.load(file)
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location=map_location)
        model = cls(**cfg["model"])
        model.load_state_dict(checkpoint["tacotron"])
        model.eval()
        return model

    def forward(self, x, mels):
        """
        Parallel forward pass for training.

        Args:
            x: text input [B, T_text]
            mels: target mel spectrograms [B, n_mels, T_mel]

        Returns:
            outputs: predicted mel spectrograms [B, n_mels, T_mel]
            alignments: attention alignments [B, T_enc, T_dec]
        """
        # Encode text
        h = self.encoder(x)  # [B, T_enc, encoder_dim]

        # Decode in parallel
        outputs, alignments = self.decoder(h, mels)

        return outputs, alignments

    def generate(self, x, max_length=10000, stop_threshold=-0.2):
        """
        Generates a log-Mel spectrogram from text.

        Parameters:
            x (Tensor): The text to synthesize converted to a sequence of symbol ids.
                See `text_to_id`.
            max_length (int): Maximum number of frames to generate.
                Defaults to 10000 frames i.e. 125 seconds.
            stop_threshold (float): If a frame is generated with all values exceeding
                `stop_threshold` then generation is stopped.

        Returns:
            Tensor: a log-Mel spectrogram of the synthesized speech.
        """
        h = self.encoder(x)
        B = x.size(0)

        # Initialize mu (GMM mean positions) to zeros
        mu = torch.zeros(B, self.n_kernels, 1, device=x.device)

        # Initialize KV cache for self-attention
        kv_cache = None

        go_frame = torch.zeros(B, self.n_mels, device=x.device)

        ys, alphas = [], []
        prev_mel = go_frame

        for t in range(0, max_length, self.reduction_factor):
            # Generate one step
            y, alpha, mu, kv_cache = self.decoder.generate_step(
                h, prev_mel, mu, kv_cache
            )

            if torch.all(y[:, :, -1] > stop_threshold):
                break

            ys.append(y)
            alphas.append(alpha)

            # Use last frame of output as next input
            prev_mel = y[:, :, -1]

        if len(ys) == 0:
            # Return empty tensors if nothing was generated
            return (
                torch.zeros(B, self.n_mels, 0, device=x.device),
                torch.zeros(B, h.size(1), 0, device=x.device),
            )

        ys = torch.cat(ys, dim=-1)
        alphas = torch.stack(alphas, dim=2)
        return ys, alphas


class GMMv2AttentionParallel(nn.Module):
    """
    Parallel GMMv2 (Gaussian Mixture Model v2) attention mechanism.

    This version processes all decoder timesteps in parallel using cumsum
    to compute the mean positions, enabling efficient parallel training.

    Based on "Location-Relative Attention Mechanisms For Robust Long-Form
    Speech Synthesis" (https://arxiv.org/abs/1910.10288)
    """

    def __init__(
        self,
        query_size,
        hidden_size,
        n_kernels,
        delta_bias,
        sigma_bias,
    ):
        super().__init__()
        self.n_kernels = n_kernels

        self.query_layer = nn.Linear(query_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 3 * n_kernels, bias=True)

        # Initialize biases for delta and sigma to encourage fast alignment
        nn.init.constant_(
            self.v.bias[n_kernels : 2 * n_kernels], delta_bias
        )
        nn.init.constant_(
            self.v.bias[2 * n_kernels : 3 * n_kernels], sigma_bias
        )

    def forward(self, queries, memory_length, mask=None):
        """
        Parallel attention computation for all decoder timesteps.

        Args:
            queries: decoder hidden states [B, T_dec, query_size]
            memory_length: length of encoder sequence (int)
            mask: optional mask for padded encoder positions [B, T_enc]

        Returns:
            alignments: attention weights [B, T_dec, T_enc]
            all_mu: all mean positions [B, T_dec, n_kernels, 1] (for visualization)
        """
        B, T_dec, _ = queries.size()
        device = queries.device

        # Compute mixture parameters from all queries at once
        # queries: [B, T_dec, query_size]
        processed_queries = self.v(torch.tanh(self.query_layer(queries)))
        # processed_queries: [B, T_dec, 3 * n_kernels]

        w_hat, delta_hat, sigma_hat = torch.chunk(processed_queries, 3, dim=-1)
        # Each: [B, T_dec, n_kernels]

        # Mixture weights (softmax over kernels for each timestep)
        w = torch.softmax(w_hat, dim=-1)  # [B, T_dec, n_kernels]

        # Step size (softplus for positivity) + small epsilon for stability
        delta = F.softplus(delta_hat) + 1e-6  # [B, T_dec, n_kernels]

        # Standard deviation (softplus for positivity)
        sigma = F.softplus(sigma_hat) + 1e-6  # [B, T_dec, n_kernels]

        # Compute all mean positions using cumulative sum (key parallelism!)
        # mu[t] = sum(delta[0:t+1]) for each kernel
        all_mu = torch.cumsum(delta, dim=1)  # [B, T_dec, n_kernels]

        # Create time indices [1, 1, 1, T_enc]
        t = torch.arange(1, memory_length + 1, device=device, dtype=queries.dtype)
        t = t.view(1, 1, 1, -1)  # [1, 1, 1, T_enc]

        # Reshape for broadcasting
        # all_mu: [B, T_dec, n_kernels] -> [B, T_dec, n_kernels, 1]
        # sigma: [B, T_dec, n_kernels] -> [B, T_dec, n_kernels, 1]
        # w: [B, T_dec, n_kernels] -> [B, T_dec, n_kernels, 1]
        all_mu_expanded = all_mu.unsqueeze(-1)  # [B, T_dec, n_kernels, 1]
        sigma_expanded = sigma.unsqueeze(-1)  # [B, T_dec, n_kernels, 1]
        w_expanded = w.unsqueeze(-1)  # [B, T_dec, n_kernels, 1]

        # Compute log energies using Gaussian pdf (in log domain for stability)
        # [B, T_dec, n_kernels, T_enc]
        z = math.sqrt(2 * math.pi) * sigma_expanded
        log_energies = -torch.log(z) - 0.5 * (t - all_mu_expanded) ** 2 / (sigma_expanded ** 2)

        # Apply mask if provided (mask padded encoder positions)
        if mask is not None:
            # mask: [B, T_enc] -> [B, 1, 1, T_enc]
            log_energies = log_energies.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e10)

        # Weighted sum of Gaussians
        energies = w_expanded * F.softmax(log_energies, dim=-1)  # [B, T_dec, n_kernels, T_enc]
        alignments = torch.sum(energies, dim=2)  # [B, T_dec, T_enc]

        return alignments, all_mu_expanded

    def forward_step(self, query, prev_mu, memory_length, mask=None):
        """
        Single-step attention for autoregressive generation.

        Args:
            query: decoder hidden state [B, query_size]
            prev_mu: previous mean positions [B, n_kernels, 1]
            memory_length: length of encoder sequence (int)
            mask: optional mask for padded positions [B, T_enc]

        Returns:
            alignments: attention weights [B, T_enc]
            current_mu: updated mean positions [B, n_kernels, 1]
        """
        device = query.device

        # Compute mixture parameters from query
        processed_query = self.v(torch.tanh(self.query_layer(query)))
        w_hat, delta_hat, sigma_hat = torch.chunk(processed_query, 3, dim=1)

        # Mixture weights (softmax over kernels)
        w = torch.softmax(w_hat, dim=1).unsqueeze(2)  # [B, n_kernels, 1]

        # Step size (softplus for positivity) + small epsilon for stability
        delta = F.softplus(delta_hat).unsqueeze(2) + 1e-6  # [B, n_kernels, 1]

        # Standard deviation (softplus for positivity)
        sigma = F.softplus(sigma_hat).unsqueeze(2) + 1e-6  # [B, n_kernels, 1]

        # Update mean position
        current_mu = prev_mu + delta  # [B, n_kernels, 1]

        # Create time indices [1, 1, T_enc]
        t = torch.arange(1, memory_length + 1, device=device, dtype=query.dtype)
        t = t.view(1, 1, -1)

        # Compute log energies using Gaussian pdf (in log domain for stability)
        z = math.sqrt(2 * math.pi) * sigma
        log_energies = -torch.log(z) - 0.5 * (t - current_mu) ** 2 / (sigma ** 2)

        # Apply mask if provided
        if mask is not None:
            log_energies = log_energies.masked_fill(mask.unsqueeze(1), -1e10)

        # Weighted sum of Gaussians
        energies = w * F.softmax(log_energies, dim=-1)  # [B, n_kernels, T_enc]
        alignments = torch.sum(energies, dim=1)  # [B, T_enc]

        return alignments, current_mu


class PreNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.5,
        fixed=False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.p = dropout
        self.fixed = fixed

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training or self.fixed)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training or self.fixed)
        return x


class BatchNormConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bnorm = nn.BatchNorm1d(output_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        g = torch.sigmoid(x2)
        return g * F.relu(x1) + (1.0 - g) * x


class CBHG(nn.Module):
    def __init__(
        self,
        K,
        input_channels,
        channels,
        projection_channels,
        n_highways,
        highway_size,
        rnn_size,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.output_size = 2 * rnn_size  # Match original bidirectional GRU output

        self.conv_bank = nn.ModuleList(
            [
                BatchNormConv(input_channels, channels, kernel_size)
                for kernel_size in range(1, K + 1)
            ]
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_projections = nn.Sequential(
            BatchNormConv(K * channels, projection_channels, 3),
            BatchNormConv(projection_channels, input_channels, 3, relu=False),
        )

        self.project = (
            nn.Linear(input_channels, highway_size, bias=False)
            if input_channels != highway_size
            else None
        )

        self.highway = nn.Sequential(
            *[HighwayNetwork(highway_size) for _ in range(n_highways)]
        )

        # Replace bidirectional GRU with self-attention layers
        self.proj_to_attn = nn.Linear(highway_size, self.output_size)
        self.pos_encoding = SinusoidalPositionalEncoding(self.output_size, dropout=dropout)
        self.self_attn_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.output_size,
                    n_heads=n_heads,
                    d_ff=self.output_size * 4,
                    dropout=dropout,
                    causal=False,  # Bidirectional attention (like bidirectional GRU)
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        T = x.size(-1)
        residual = x

        x = [conv(x)[:, :, :T] for conv in self.conv_bank]
        x = torch.cat(x, dim=1)

        x = self.max_pool(x)

        x = self.conv_projections(x[:, :, :T])

        x = x + residual
        x = x.transpose(1, 2)  # [B, T, C]

        if self.project is not None:
            x = self.project(x)

        x = self.highway(x)

        # Self-attention layers (replacing bidirectional GRU)
        x = self.proj_to_attn(x)
        x = self.pos_encoding(x)
        for layer in self.self_attn_layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_symbols, embedding_dim, prenet, cbhg):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, embedding_dim)
        self.pre_net = PreNet(**prenet)
        self.cbhg = CBHG(**cbhg)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pre_net(x)
        x = self.cbhg(x.transpose(1, 2))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for attending to encoder outputs."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, memory_key_padding_mask=None):
        """
        Args:
            x: decoder hidden states [B, T_dec, D]
            memory: encoder outputs [B, T_enc, D]
            memory_key_padding_mask: [B, T_enc] mask for encoder padding
        Returns:
            [B, T_dec, D] tensor
        """
        attn_out, _ = self.cross_attn(
            x, memory, memory, key_padding_mask=memory_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class Decoder(nn.Module):
    """
    Parallel decoder using self-attention (replacing LSTMs).

    For training, processes all mel frames in parallel with causal masking.
    For generation, uses step-by-step decoding with KV cache.
    """

    def __init__(
        self,
        prenet,
        attention,
        input_size,
        n_mels,
        attn_rnn_size,  # Now used as decoder hidden size
        decoder_rnn_size,  # Also decoder hidden size (should match attn_rnn_size)
        reduction_factor,
        zoneout_prob=0.1,  # Kept for config compatibility, but dropout used instead
        n_layers=3,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.reduction_factor = reduction_factor
        self.n_kernels = attention["n_kernels"]
        self.hidden_size = decoder_rnn_size
        encoder_dim = 2 * input_size  # Encoder output dimension

        # PreNet for mel inputs
        self.prenet = PreNet(**prenet)

        # Project prenet output to decoder hidden size
        self.input_proj = nn.Linear(prenet["output_size"], self.hidden_size)

        # Positional encoding for decoder
        self.pos_encoding = SinusoidalPositionalEncoding(self.hidden_size, dropout=dropout)

        # Causal self-attention layers
        self.self_attn_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.hidden_size,
                    n_heads=n_heads,
                    d_ff=self.hidden_size * 4,
                    dropout=dropout,
                    causal=True,  # Causal masking for autoregressive decoding
                )
                for _ in range(n_layers)
            ]
        )

        # Cross-attention to encoder outputs (after self-attention)
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=self.hidden_size,
                    n_heads=n_heads,
                    d_ff=self.hidden_size * 4,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # GMM attention for learning alignments (uses cumsum parallelism)
        # We need to rename attn_rnn_size to query_size for the new attention
        attn_config = dict(attention)
        attn_config["query_size"] = attn_config.pop("attn_rnn_size")
        self.gmm_attention = GMMv2AttentionParallel(**attn_config)

        # Project encoder outputs to decoder dimension if needed
        self.encoder_proj = (
            nn.Linear(encoder_dim, self.hidden_size)
            if encoder_dim != self.hidden_size
            else nn.Identity()
        )

        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, n_mels * reduction_factor)

    def forward(self, encoder_outputs, mels):
        """
        Parallel forward pass for training.

        Args:
            encoder_outputs: [B, T_enc, encoder_dim]
            mels: [B, n_mels, T_mel] target mel spectrograms

        Returns:
            outputs: [B, n_mels, T_mel] predicted mel spectrograms
            alignments: [B, T_dec, T_enc] attention alignments
        """
        B, N, T = mels.size()
        T_enc = encoder_outputs.size(1)

        # Project encoder outputs
        memory = self.encoder_proj(encoder_outputs)  # [B, T_enc, hidden_size]

        # Prepare decoder inputs (teacher forcing with go frame)
        # Take every reduction_factor-th frame and shift right
        T_dec = T // self.reduction_factor
        # mels: [B, n_mels, T] -> sample at reduction_factor intervals
        mel_inputs = mels[:, :, ::self.reduction_factor]  # [B, n_mels, T_dec]
        # Shift right: prepend go frame, remove last
        go_frame = torch.zeros(B, N, 1, device=mels.device)
        mel_inputs = torch.cat([go_frame, mel_inputs[:, :, :-1]], dim=2)  # [B, n_mels, T_dec]
        mel_inputs = mel_inputs.transpose(1, 2)  # [B, T_dec, n_mels]

        # PreNet
        x = self.prenet(mel_inputs.reshape(B * T_dec, N))
        x = x.view(B, T_dec, -1)  # [B, T_dec, prenet_output_size]

        # Project to hidden size and add positional encoding
        x = self.input_proj(x)  # [B, T_dec, hidden_size]
        x = self.pos_encoding(x)

        # Self-attention and cross-attention layers
        for self_attn, cross_attn in zip(self.self_attn_layers, self.cross_attn_layers):
            x = self_attn(x)
            x = cross_attn(x, memory)

        # GMM attention (parallel with cumsum)
        alignments, _ = self.gmm_attention(x, T_enc)  # [B, T_dec, T_enc]

        # Attend to encoder outputs using GMM alignments
        context = torch.bmm(alignments, encoder_outputs)  # [B, T_dec, encoder_dim]
        context = self.encoder_proj(context)  # [B, T_dec, hidden_size]

        # Combine with decoder states and project to output
        x = x + context  # Residual connection with context
        outputs = self.output_proj(x)  # [B, T_dec, n_mels * reduction_factor]

        # Reshape outputs
        outputs = outputs.view(B, T_dec, N, self.reduction_factor)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()  # [B, n_mels, T_dec, rf]
        outputs = outputs.view(B, N, -1)  # [B, n_mels, T]

        # Transpose alignments for compatibility: [B, T_dec, T_enc] -> [B, T_enc, T_dec]
        alignments = alignments.transpose(1, 2)

        return outputs, alignments

    def generate_step(self, encoder_outputs, prev_mel, mu, kv_cache=None):
        """
        Single step generation for autoregressive inference.

        Args:
            encoder_outputs: [B, T_enc, encoder_dim]
            prev_mel: [B, n_mels] previous mel frame
            mu: [B, n_kernels, 1] previous GMM mean positions
            kv_cache: list of cached key-value pairs for each layer

        Returns:
            output: [B, n_mels, reduction_factor] predicted mel frames
            alpha: [B, T_enc] attention weights
            mu: [B, n_kernels, 1] updated GMM mean positions
            kv_cache: updated cache
        """
        B = prev_mel.size(0)
        T_enc = encoder_outputs.size(1)

        # Project encoder outputs
        memory = self.encoder_proj(encoder_outputs)

        # PreNet
        x = self.prenet(prev_mel)  # [B, prenet_output_size]
        x = self.input_proj(x)  # [B, hidden_size]
        x = x.unsqueeze(1)  # [B, 1, hidden_size]

        # Get current position for positional encoding
        if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
            pos = kv_cache[0][0].size(1)  # Number of cached positions
        else:
            pos = 0

        # Add positional encoding for current position
        x = x + self.pos_encoding.pe[:, pos : pos + 1]

        # Initialize cache if needed
        if kv_cache is None:
            kv_cache = [None] * len(self.self_attn_layers)

        new_kv_cache = []
        for i, (self_attn, cross_attn) in enumerate(
            zip(self.self_attn_layers, self.cross_attn_layers)
        ):
            # Self-attention with KV cache
            if kv_cache[i] is not None:
                cached_k, cached_v = kv_cache[i]
                # Append current k, v to cache
                k = torch.cat([cached_k, x], dim=1)
                v = torch.cat([cached_v, x], dim=1)
            else:
                k = v = x

            # Compute attention (only for current query position)
            attn_out, _ = self_attn.self_attn(x, k, v)
            x = self_attn.norm1(x + self_attn.dropout(attn_out))
            x = self_attn.norm2(x + self_attn.ff(x))

            # Cross-attention
            x = cross_attn(x, memory)

            # Store updated cache
            new_kv_cache.append((k, v))

        # GMM attention (single step)
        query = x.squeeze(1)  # [B, hidden_size]
        alpha, mu = self.gmm_attention.forward_step(query, mu, T_enc)

        # Attend to encoder outputs
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs)  # [B, 1, encoder_dim]
        context = self.encoder_proj(context)  # [B, 1, hidden_size]

        # Combine and project
        x = x + context
        output = self.output_proj(x)  # [B, 1, n_mels * reduction_factor]
        output = output.view(B, self.n_mels, self.reduction_factor)

        return output, alpha, mu, new_kv_cache
