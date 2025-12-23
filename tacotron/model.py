import importlib_resources
import math

import toml
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tacotron(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.input_size = 2 * decoder["input_size"]
        self.attn_rnn_size = decoder["attn_rnn_size"]
        self.decoder_rnn_size = decoder["decoder_rnn_size"]
        self.n_mels = decoder["n_mels"]
        self.reduction_factor = decoder["reduction_factor"]
        self.n_kernels = decoder["attention"]["n_kernels"]

        self.encoder = Encoder(**encoder)
        self.decoder_cell = DecoderCell(**decoder)

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
        B, N, T = mels.size()
        mels = mels.unbind(-1)

        h = self.encoder(x)

        # Initialize mu (GMM mean positions) to zeros
        mu = torch.zeros(B, self.n_kernels, 1, device=x.device)
        c = torch.zeros(B, self.input_size, device=x.device)

        attn_hx = (
            torch.zeros(B, self.attn_rnn_size, device=x.device),
            torch.zeros(B, self.attn_rnn_size, device=x.device),
        )

        rnn1_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
        )

        rnn2_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
        )

        go_frame = torch.zeros(B, N, device=x.device)

        ys, alphas = [], []
        for t in range(0, T, self.reduction_factor):
            y = mels[t - 1] if t > 0 else go_frame
            y, alpha, mu, c, attn_hx, rnn1_hx, rnn2_hx = self.decoder_cell(
                h, y, mu, c, attn_hx, rnn1_hx, rnn2_hx
            )
            ys.append(y)
            alphas.append(alpha)

        ys = torch.cat(ys, dim=-1)
        alphas = torch.stack(alphas, dim=2)
        return ys, alphas

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
        B, T, _ = h.size()

        # Initialize mu (GMM mean positions) to zeros
        mu = torch.zeros(B, self.n_kernels, 1, device=x.device)
        c = torch.zeros(B, self.input_size, device=x.device)

        attn_hx = (
            torch.zeros(B, self.attn_rnn_size, device=x.device),
            torch.zeros(B, self.attn_rnn_size, device=x.device),
        )

        rnn1_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
        )

        rnn2_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
            torch.zeros(B, self.decoder_rnn_size, device=x.device),
        )

        go_frame = torch.zeros(B, self.n_mels, device=x.device)

        ys, alphas = [], []
        for t in range(0, max_length, self.reduction_factor):
            y = ys[-1][:, :, -1] if t > 0 else go_frame
            y, alpha, mu, c, attn_hx, rnn1_hx, rnn2_hx = self.decoder_cell(
                h, y, mu, c, attn_hx, rnn1_hx, rnn2_hx
            )
            if torch.all(y[:, :, -1] > stop_threshold):
                break
            ys.append(y)
            alphas.append(alpha)

        ys = torch.cat(ys, dim=-1)
        alphas = torch.stack(alphas, dim=2)
        return ys, alphas


class GMMv2Attention(nn.Module):
    """
    GMMv2 (Gaussian Mixture Model v2) attention mechanism from
    "Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis"
    (https://arxiv.org/abs/1910.10288)

    Uses a mixture of Gaussians to model attention, with the mean position
    updated incrementally at each step. This is a location-relative mechanism
    that generalizes well to long-form synthesis.
    """

    def __init__(
        self,
        attn_rnn_size,
        hidden_size,
        n_kernels,
        delta_bias,
        sigma_bias,
    ):
        super(GMMv2Attention, self).__init__()
        self.n_kernels = n_kernels

        self.query_layer = nn.Linear(attn_rnn_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 3 * n_kernels, bias=True)

        # Initialize biases for delta and sigma to encourage fast alignment
        nn.init.constant_(
            self.v.bias[n_kernels : 2 * n_kernels], delta_bias
        )
        nn.init.constant_(
            self.v.bias[2 * n_kernels : 3 * n_kernels], sigma_bias
        )

    def forward(self, query, prev_mu, memory_length, mask=None):
        """
        Args:
            query: decoder hidden state [B, attn_rnn_size]
            prev_mu: previous mean positions [B, n_kernels, 1]
            memory_length: length of encoder sequence (int)
            mask: optional mask for padded positions [B, T]

        Returns:
            alignments: attention weights [B, T]
            current_mu: updated mean positions [B, n_kernels, 1]
        """
        B = query.size(0)
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

        # Create time indices [1, 1, T]
        t = torch.arange(1, memory_length + 1, device=device).float()
        t = t.view(1, 1, -1)

        # Compute log energies using Gaussian pdf (in log domain for stability)
        # log(w * N(t; mu, sigma)) = log(w) + log(N(t; mu, sigma))
        z = math.sqrt(2 * math.pi) * sigma
        log_energies = -torch.log(z) - 0.5 * (t - current_mu) ** 2 / (sigma ** 2)

        # Apply mask if provided
        if mask is not None:
            log_energies = log_energies.masked_fill(mask.unsqueeze(1), -1e10)

        # Weighted sum of Gaussians
        energies = w * F.softmax(log_energies, dim=-1)  # [B, n_kernels, T]
        alignments = torch.sum(energies, dim=1)  # [B, T]

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
    ):
        super().__init__()

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

        self.rnn = nn.GRU(highway_size, rnn_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        T = x.size(-1)
        residual = x

        x = [conv(x)[:, :, :T] for conv in self.conv_bank]
        x = torch.cat(x, dim=1)

        x = self.max_pool(x)

        x = self.conv_projections(x[:, :, :T])

        x = x + residual
        x = x.transpose(1, 2)

        if self.project is not None:
            x = self.project(x)

        x = self.highway(x)

        x, _ = self.rnn(x)
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


def zoneout(prev, current, p=0.1):
    mask = torch.empty_like(prev).bernoulli_(p)
    return mask * prev + (1 - mask) * current


class DecoderCell(nn.Module):
    def __init__(
        self,
        prenet,
        attention,
        input_size,
        n_mels,
        attn_rnn_size,
        decoder_rnn_size,
        reduction_factor,
        zoneout_prob,
    ):
        super(DecoderCell, self).__init__()
        self.zoneout_prob = zoneout_prob
        self.n_kernels = attention["n_kernels"]

        self.prenet = PreNet(**prenet)
        self.attention = GMMv2Attention(**attention)
        self.attn_rnn = nn.LSTMCell(
            2 * input_size + prenet["output_size"], attn_rnn_size
        )
        self.linear = nn.Linear(2 * input_size + decoder_rnn_size, decoder_rnn_size)
        self.rnn1 = nn.LSTMCell(decoder_rnn_size, decoder_rnn_size)
        self.rnn2 = nn.LSTMCell(decoder_rnn_size, decoder_rnn_size)
        self.proj = nn.Linear(decoder_rnn_size, n_mels * reduction_factor, bias=False)

    def forward(self, h, y, mu, c, attn_hx, rnn1_hx, rnn2_hx):
        B, N = y.size()
        T = h.size(1)

        y = self.prenet(y)
        attn_h, attn_c = self.attn_rnn(torch.cat((c, y), dim=-1), attn_hx)
        if self.training:
            attn_h = zoneout(attn_hx[0], attn_h, p=self.zoneout_prob)

        alpha, mu = self.attention(attn_h, mu, T)

        c = torch.matmul(alpha.unsqueeze(1), h).squeeze(1)

        x = self.linear(torch.cat((c, attn_h), dim=-1))

        rnn1_h, rnn1_c = self.rnn1(x, rnn1_hx)
        if self.training:
            rnn1_h = zoneout(rnn1_hx[0], rnn1_h, p=self.zoneout_prob)
        x = x + rnn1_h

        rnn2_h, rnn2_c = self.rnn2(x, rnn2_hx)
        if self.training:
            rnn2_h = zoneout(rnn2_hx[0], rnn2_h, p=self.zoneout_prob)
        x = x + rnn2_h

        y = self.proj(x).view(B, N, 2)
        return y, alpha, mu, c, (attn_h, attn_c), (rnn1_h, rnn1_c), (rnn2_h, rnn2_c)
