import torch
from torch import nn
from collections import deque

class gcntf(torch.nn.Module):
    class DecoderZH(torch.nn.Module):
        def __init__(self, z_dim, hidden_dim, embed_dim, output_dim):
            super().__init__()
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(z_dim + hidden_dim, embed_dim),
                torch.nn.ReLU6(),
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU6()
            )
            self.mu = torch.nn.Linear(embed_dim, output_dim)

        def forward(self, z, h):
            dim_0 = h.size(0)
            dim_1 = h.size(1)
            if dim_0 > 256:
                h = h[0:256, :]
            if dim_1 > 256:
                h = h[:, 0:256]
            xy = self.embed(torch.cat((z, h), -1))
            loc = self.mu(xy)
            return loc

    class P_Z(torch.nn.Module):
        def __init__(self, hidden_dim_fy, embed_dim, z_dim):
            super().__init__()
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim_fy, embed_dim),
                torch.nn.ReLU6(),
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU6()
            )
            self.mu = torch.nn.Linear(embed_dim, z_dim)
            self.std = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, z_dim),
                torch.nn.Softplus()
            )

        def forward(self, x):
            dim_0 = x.size(0)
            dim_1 = x.size(1)
            if dim_0 > 256:
                x = x[0:256, :]
            if dim_1 > 256:
                x = x[:, 0:256]
            x = self.embed(x)
            loc = self.mu(x)
            std = self.std(x)
            return torch.distributions.Normal(loc, std)

    class Q_Z(torch.nn.Module):
        def __init__(self, hidden_dim_fy, hidden_dim_by, embed_dim, z_dim):
            super().__init__()
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim_fy + hidden_dim_by, embed_dim),
                torch.nn.ReLU6(),
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU6()
            )
            self.mu = torch.nn.Linear(embed_dim, z_dim)
            self.std = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, z_dim),
                torch.nn.Softplus()
            )

        def forward(self, x, y):
            xy = self.embed(torch.cat((x, y), -1))
            loc = self.mu(xy)
            std = self.std(xy)
            return torch.distributions.Normal(loc, std)

    class EmbedZD(torch.nn.Module):
        def __init__(self, z_dim, d_dim, output_dim):
            super().__init__()
            self.embed_zd = torch.nn.Sequential(
                torch.nn.Linear(z_dim + d_dim, output_dim),
                torch.nn.ReLU6(),
                torch.nn.Linear(output_dim, output_dim)
            )

        def forward(self, z, d):
            code = torch.cat((z, d), -1)
            return self.embed_zd(code)

    def __init__(self, horizon, ob_radius=2, hidden_dim=256):
        super().__init__()
        self.ob_radius = ob_radius
        self.horizon = horizon
        hidden_dim_fx = hidden_dim
        hidden_dim_fy = hidden_dim
        hidden_dim_by = 256
        feature_dim = 256
        self_embed_dim = 128
        neighbor_embed_dim = 128
        z_dim = 32
        d_dim = 2

        self.q_z = gcntf.Q_Z(hidden_dim_fy, hidden_dim_by, hidden_dim_fy, z_dim)
        self.p_z = gcntf.P_Z(hidden_dim_fy, hidden_dim_fy, z_dim)
        self.dec = gcntf.DecoderZH(z_dim, hidden_dim_fy, hidden_dim_fy, d_dim)
        self.gc_weight = torch.nn.Parameter(torch.randn(hidden_dim_fx, hidden_dim_fx))
        self.gc_bias = torch.nn.Parameter(torch.zeros(hidden_dim_fx))


        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.st_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.feature_transform = nn.Linear(6, hidden_dim)
        self.gcn_outputs_queue = deque(maxlen=8)
        self.dim_reduction = nn.Linear(hidden_dim * 3, hidden_dim)
        self.zd_to_hidden = nn.Linear(in_features=32, out_features=hidden_dim)
        self.shape_adjust_linear = nn.Linear(1024, 128)

        self.embed_s = torch.nn.Sequential(
            torch.nn.Linear(4, 64),  # v, a
            torch.nn.ReLU6(),
            torch.nn.Linear(64, self_embed_dim),
        )
        self.embed_n = torch.nn.Sequential(
            torch.nn.Linear(4, 64),  # dp, dv
            torch.nn.ReLU6(),
            torch.nn.Linear(64, neighbor_embed_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(neighbor_embed_dim, neighbor_embed_dim)
        )
        self.embed_k = torch.nn.Sequential(
            torch.nn.Linear(3, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim)
        )
        self.embed_q = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_fx, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim)
        )
        self.attention_nonlinearity = torch.nn.LeakyReLU(0.2)

        self.rnn_fx = torch.nn.GRU(self_embed_dim + neighbor_embed_dim, hidden_dim_fx)
        self.rnn_fx_init = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim_fx),  # dp
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_dim_fx, hidden_dim_fx * self.rnn_fx.num_layers),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_dim_fx * self.rnn_fx.num_layers, hidden_dim_fx * self.rnn_fx.num_layers),
        )
        self.rnn_by = torch.nn.GRU(self_embed_dim + neighbor_embed_dim, hidden_dim_by)

        self.embed_zd = gcntf.EmbedZD(z_dim, d_dim, z_dim)
        self.rnn_fy = torch.nn.GRU(z_dim, hidden_dim_fy)
        self.rnn_fy_init = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_fx, hidden_dim_fy * self.rnn_fy.num_layers),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_dim_fy * self.rnn_fy.num_layers, hidden_dim_fy * self.rnn_fy.num_layers)
        )

    def attention(self, q, k, mask):
        # q: N x d
        # k: N x Nn x d
        # mask: N x Nn
        e = (k @ q.unsqueeze(-1)).squeeze(-1)  # N x Nn
        e = self.attention_nonlinearity(e)  # N x Nn
        e[~mask] = -float("inf")
        att = torch.nn.functional.softmax(e, dim=-1)  # N x Nn
        return att.nan_to_num()

    def gcn_layer(self, input, similarity):
        if isinstance(similarity, list) and len(similarity) == 1:
            similarity = similarity[0]
        elif isinstance(similarity, torch.Tensor) and similarity.numel() == 1:
            similarity = similarity.item()
        similarity = torch.tensor(similarity, dtype=torch.float32)
        weights = 1 - similarity
        support = torch.einsum('bik,kj->bij', input, self.gc_weight[:input.size(-1), :input.size(-1)])

        if self.gc_bias is not None:
            bias_shape = self.gc_bias.size()
            if bias_shape[0] != input.size(-1):
                new_gc_bias_data = self.gc_bias[:input.size(-1)].detach().clone()
                self.gc_bias.data = new_gc_bias_data.requires_grad_(True)
            support = support + self.gc_bias.unsqueeze(0).unsqueeze(0)
        weights = weights.mean(dim=-1, keepdim=True)
        output = support * weights

        return output


    def enc(self, x, neighbor, *, y=None, similarity=None):
        # x: (L1+1) x N x 6
        # y: L2 x N x 2
        # neighbor: (L1+L2+1) x N x Nn x 6
        global h_t
        with torch.no_grad():
            L1 = x.size(0) - 1
            N = neighbor.size(1)
            Nn = neighbor.size(2)
            state = x

            x = state[..., :2]  # (L1+1) x N x 2
            if y is not None:
                L2 = y.size(0)
                x = torch.cat((x, y), 0)  # (L+1) x N x 2
            else:
                L2 = 0

            v = x[1:] - x[:-1]  # L x N x 2
            a = v[1:] - v[:-1]  # (L-1) x N x 2
            a = torch.cat((state[1:2, ..., 4:6], a))  # L x N x 2

            neighbor_x = neighbor[..., :2]  # (L+1) x N x Nn x 2
            neighbor_v = neighbor[1:, ..., 2:4]  # L x N x Nn x 2

            dp = neighbor_x - x.unsqueeze(-2)  # (L+1) x N x Nn x 2
            dv = neighbor_v - v.unsqueeze(-2)  # L x N x Nn x 2

            # social features
            dist = dp.norm(dim=-1)  # (L+1) x N x Nn
            mask = dist <= self.ob_radius
            dp0, mask0 = dp[0], mask[0]
            dp, mask = dp[1:], mask[1:]
            dist = dist[1:]
            dot_dp_v = (dp @ v.unsqueeze(-1)).squeeze(-1)  # L x N x Nn
            bearing = dot_dp_v / (dist * v.norm(dim=-1).unsqueeze(-1))  # L x N x Nn
            bearing = bearing.nan_to_num(0, 0, 0)
            dot_dp_dv = (dp.unsqueeze(-2) @ dv.unsqueeze(-1)).view(dp.size(0), N, Nn)
            tau = -dot_dp_dv / dv.norm(dim=-1)  # L x N x Nn
            tau = tau.nan_to_num(0, 0, 0).clip(0, 7)
            mpd = (dp + tau.unsqueeze(-1) * dv).norm(dim=-1)  # L x N x Nn
            features = torch.stack((dist, bearing, mpd), -1)  # L x N x Nn x 3

        k = self.embed_k(features)  # L x N x Nn x d
        s = self.embed_s(torch.cat((v, a), -1))
        n = self.embed_n(torch.cat((dp, dv), -1))  # L x N x Nn x ...

        h = self.rnn_fx_init(dp0)  # N x Nn x d
        h = (mask0.unsqueeze(-1) * h).sum(-2)  # N x d
        h = h.view(N, -1, self.rnn_fx.num_layers)
        h = h.permute(2, 0, 1).contiguous()

        trajectory_features = state
        trajectory_features = self.feature_transform(trajectory_features)
        batch_size, seq_len, feature_dim = trajectory_features.shape
        trajectory_features = trajectory_features.transpose(0, 1)
        attn_output, _ = self.self_attn(trajectory_features, trajectory_features, trajectory_features)
        attn_output = attn_output.transpose(0, 1)


        gcn_output = self.gcn_layer(attn_output, similarity)
        self.gcn_outputs_queue.append(gcn_output)

        if len(self.gcn_outputs_queue) == self.gcn_outputs_queue.maxlen:
            spatio_temporal_features = torch.stack(list(self.gcn_outputs_queue), dim=2)
            st_attn_input = spatio_temporal_features.view(-1, spatio_temporal_features.size(2), feature_dim)
            st_attn_output, _ = self.st_attn(st_attn_input, st_attn_input, st_attn_input)
            st_attn_output = st_attn_output.view(batch_size, seq_len, -1, feature_dim).mean(dim=2)
            attn_output = attn_output.view(batch_size, seq_len, -1)
            spatio_temporal_mean = spatio_temporal_features.mean(dim=2).view(batch_size, seq_len, -1)
            combined_features = torch.cat((attn_output, spatio_temporal_mean, st_attn_output),dim=-1)
            transformed_features = self.dim_reduction(combined_features)
            combined_tensor = torch.cat([h, transformed_features], dim=0)
            result_tensor = combined_tensor[0].unsqueeze(0)
            h = result_tensor
        for t in range(L1):
            q = self.embed_q(h[-1])  # N x d
            att = self.attention(q, k[t], mask[t])  # N x Nn
            x_t = att.unsqueeze(-2) @ n[t]  # N x 1 x d
            x_t = x_t.squeeze(-2)  # N x d
            x_t = torch.cat((x_t, s[t]), -1).unsqueeze(0)
            _, h = self.rnn_fx(x_t, h)
        x = h[-1]
        if y is None: return x
        mask_t = mask[L1:L1 + L2].unsqueeze(-1)  # L2 x N x Nn x 1
        n_t = n[L1:L1 + L2]  # L2 x N x Nn x d
        n_t = (mask_t * n_t).sum(-2)  # L2 x N x d
        s_t = s[L1:L2 + L2]
        x_t = torch.cat((n_t, s_t), -1)
        x_t = torch.flip(x_t, (0,))
        b, _ = self.rnn_by(x_t)  # L2 x N x n_layer*d
        if self.rnn_by.num_layers > 1:
            b = b[..., -b.size(-1) // self.rnn_by.num_layers:]
        b = torch.flip(b, (0,))
        return x, b

    def forward(self, *args, **kwargs):
        self.rnn_fx.flatten_parameters()
        self.rnn_fy.flatten_parameters()
        sim = []
        if self.training:
            self.rnn_by.flatten_parameters()
            args = iter(args)
            x = kwargs["x"] if "x" in kwargs else next(args)
            y = kwargs["y"] if "y" in kwargs else next(args)
            neighbor = kwargs["neighbor"] if "neighbor" in kwargs else next(args)
            similarity = kwargs["SIMILARITY"] if "SIMILARITY" in kwargs else next(args)
            if similarity is None:
                sim.append(0)
            else:
                sim.append(similarity)
            return self.learn(x, y, neighbor, similarity=sim)
        args = iter(args)
        x = kwargs["x"] if "x" in kwargs else next(args)
        neighbor = kwargs["neighbor"] if "neighbor" in kwargs else next(args)
        try:
            n_predictions = kwargs["n_predictions"] if "n_predictions" in kwargs else next(args)
            similarity = kwargs["SIMILARITY"] if "SIMILARITY" in kwargs else next(args)
        except:
            n_predictions = 0
            similarity = 0
        stochastic = n_predictions > 0
        if neighbor is None:
            neighbor_shape = [_ for _ in x.shape]
            neighbor_shape.insert(-1, 0)
            neighbor = torch.empty(neighbor_shape, dtype=x.dtype, device=x.device)
        C = x.dim()
        if C < 3:
            x = x.unsqueeze(1)
            neighbor = neighbor.unsqueeze(1)
            #if y is not None: y = y.unsqueeze(1)
        N = x.size(1)
        neighbor = neighbor[:x.size(0)]


        h = self.enc(x, neighbor,similarity=similarity)
        h = self.rnn_fy_init(h)
        h = h.view(N, -1, self.rnn_fy.num_layers)
        h = h.permute(2, 0, 1)
        if stochastic: h = h.repeat(1, n_predictions, 1)
        h = h.contiguous()
        D = []
        for t in range(self.horizon):
            p_z = self.p_z(h[-1])
            if stochastic:
                z = p_z.sample()
            else:
                z = p_z.mean
            d = self.dec(z, h[-1])
            D.append(d)
            if t == self.horizon - 1: break
            zd = self.embed_zd(z, d)
            h = h[:, :256, :]
            h = h[:, :, :256]
            _, h = self.rnn_fy(zd.unsqueeze(0), h)

        d = torch.stack(D)
        pred = torch.cumsum(d, 0)
        if stochastic:
            new_n_predictions = pred.size(1)
            pred = pred.view(pred.size(0), new_n_predictions, -1, pred.size(-1)).permute(1, 0, 2, 3)
        pred = pred + x[-1, ..., :2]
        if C < 3:
            pred = pred.squeeze(1)
        return pred

    def learn(self, x, y, neighbor=None, similarity=None):
        C = x.dim()
        if C < 3:
            x = x.unsqueeze(1)
            neighbor = neighbor.unsqueeze(1)
            if y is not None: y = y.unsqueeze(1)
        N = x.size(1)
        if y.size(0) != self.horizon:
            print("[Warn] Unmatched sequence length in inference and generative model. ({} vs {})".format(y.size(0),
                                                                                                          self.horizon))
        h, b = self.enc(x, neighbor, y=y, similarity=similarity)
        h = self.rnn_fy_init(h)
        h = h.view(N, -1, self.rnn_fy.num_layers)
        h = h.permute(2, 0, 1).contiguous()

        P, Q = [], []
        D, Z = [], []
        for t in range(self.horizon):
            p_z = self.p_z(h[-1])
            q_z = self.q_z(h[-1], b[t])
            z = q_z.rsample()
            d = self.dec(z, h[-1])

            P.append(p_z)
            Q.append(q_z)
            D.append(d)
            Z.append(z)

            if t == self.horizon - 1: break
            zd = self.embed_zd(z, d)
            _, h = self.rnn_fy(zd.unsqueeze(0), h)

        d = torch.stack(D)
        with torch.no_grad():
            y = y - x[-1, ..., :2].unsqueeze(0)
        pred = torch.cumsum(d, 0)

        err = (pred - y).square()
        kl = []
        for p, q, z in zip(P, Q, Z):
            kl.append(q.log_prob(z) - p.log_prob(z))
        kl = torch.stack(kl)
        return err, kl

    def loss(self, err, kl):
        rec = err.mean()
        kl = kl.mean()

        return {
            "loss": kl + rec,
            "rec": rec,
            "kl": kl
        }