import torch
import torch.nn as nn
import torch.nn.functional as F

class gcntf(nn.Module):
    class DecoderZH(nn.Module):
        def __init__(self, z_dim, hidden_dim, embed_dim, output_dim):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(z_dim + hidden_dim, embed_dim),
                nn.ReLU6(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU6()
            )
            self.mu = nn.Linear(embed_dim, output_dim)

        def forward(self, z, h):
            xy = self.embed(torch.cat((z, h), -1))
            loc = self.mu(xy)
            return loc

    class P_Z(nn.Module):
        def __init__(self, hidden_dim_fy, embed_dim, z_dim):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(hidden_dim_fy, embed_dim),
                nn.ReLU6(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU6()
            )
            self.mu = nn.Linear(embed_dim, z_dim)
            self.std = nn.Sequential(
                nn.Linear(embed_dim, z_dim),
                nn.Softplus()
            )

        def forward(self, x):
            x = self.embed(x)
            loc = self.mu(x)
            std = self.std(x)
            return torch.distributions.Normal(loc, std)

    class Q_Z(nn.Module):
        def __init__(self, hidden_dim_fy, hidden_dim_by, embed_dim, z_dim):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(hidden_dim_fy + hidden_dim_by, embed_dim),
                nn.ReLU6(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU6()
            )
            self.mu = nn.Linear(embed_dim, z_dim)
            self.std = nn.Sequential(
                nn.Linear(embed_dim, z_dim),
                nn.Softplus()
            )

        def forward(self, x, y):
            xy = self.embed(torch.cat((x, y), -1))
            loc = self.mu(xy)
            std = self.std(xy)
            return torch.distributions.Normal(loc, std)

    class EmbedZD(nn.Module):
        def __init__(self, z_dim, d_dim, output_dim):
            super().__init__()
            self.embed_zd = nn.Sequential(
                nn.Linear(z_dim + d_dim, output_dim),
                nn.ReLU6(),
                nn.Linear(output_dim, output_dim)
            )

        def forward(self, z, d):
            code = torch.cat((z, d), -1)
            return self.embed_zd(code)

    def __init__(self, horizon, ob_radius=2, hidden_dim=256, num_gcn_layers=2):
        super().__init__()
        self.ob_radius = ob_radius
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers

        hidden_dim_fx = hidden_dim
        hidden_dim_fy = hidden_dim
        hidden_dim_by = 256
        feature_dim = 256
        self_embed_dim = 128
        neighbor_embed_dim = 128
        z_dim = 32
        d_dim = 2

        self.q_z = self.Q_Z(hidden_dim_fy, hidden_dim_by, hidden_dim_fy, z_dim)
        self.p_z = self.P_Z(hidden_dim_fy, hidden_dim_fy, z_dim)
        self.dec = self.DecoderZH(z_dim, hidden_dim_fy, hidden_dim_fy, d_dim)

        self.embed_s = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU6(),
            nn.Linear(64, self_embed_dim),
        )
        self.embed_n = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU6(),
            nn.Linear(64, neighbor_embed_dim),
            nn.ReLU6(),
            nn.Linear(neighbor_embed_dim, neighbor_embed_dim)
        )
        self.embed_k = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.embed_q = nn.Sequential(
            nn.Linear(hidden_dim_fx, feature_dim),
            nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU6(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.attention_nonlinearity = nn.LeakyReLU(0.2)

        # GCN
        self.gcn_layers = nn.ModuleList([
            nn.Linear(neighbor_embed_dim, neighbor_embed_dim) for _ in range(num_gcn_layers)
        ])

    def gcn_forward(self, n, adj_matrix):
        for layer in self.gcn_layers:
            n = torch.matmul(adj_matrix, n)
            n = layer(n)
            n = F.relu6(n)
        return n

    def attention(self, q, k, mask):
        e = (k @ q.unsqueeze(-1)).squeeze(-1)
        e = self.attention_nonlinearity(e)
        e[~mask] = -float("inf")
        att = F.softmax(e, dim=-1)
        return att.nan_to_num()

    def enc(self, x, neighbor, *, y=None):
        with torch.no_grad():
            L1 = x.size(0) - 1
            N = neighbor.size(1)
            Nn = neighbor.size(2)
            state = x

            x = state[..., :2]
            if y is not None:
                L2 = y.size(0)
                x = torch.cat((x, y), 0)
            else:
                L2 = 0

            v = x[1:] - x[:-1]
            a = v[1:] - v[:-1]
            a = torch.cat((state[1:2, ..., 4:6], a))
            neighbor_x = neighbor[..., :2]
            neighbor_v = neighbor[1:, ..., 2:4]
            dp = neighbor_x - x.unsqueeze(-2)
            dv = neighbor_v - v.unsqueeze(-2)
            dist = dp.norm(dim=-1)
            mask = dist <= self.ob_radius
            dp0, mask0 = dp[0], mask[0]
            dp, mask = dp[1:], mask[1:]
            dist = dist[1:]
            dot_dp_v = (dp @ v.unsqueeze(-1)).squeeze(-1)
            bearing = dot_dp_v / (dist * v.norm(dim=-1).unsqueeze(-1))
            bearing = bearing.nan_to_num(0, 0, 0)
            dot_dp_dv = (dp.unsqueeze(-2) @ dv.unsqueeze(-1)).view(dp.size(0), N, Nn)
            tau = -dot_dp_dv / dv.norm(dim=-1)
            tau = tau.nan_to_num(0, 0, 0).clip(0, 7)
            mpd = (dp + tau.unsqueeze(-1) * dv).norm(dim=-1)
            features = torch.stack((dist, bearing, mpd), -1)

        k = self.embed_k(features)
        s = self.embed_s(torch.cat((v, a), -1))
        n = self.embed_n(torch.cat((dp, dv), -1))

        adj_matrix = (dist <= self.ob_radius).float()
        n = self.gcn_forward(n, adj_matrix)

        h = []
        for t in range(L1):
            q = self.embed_q(s[t])
            att = self.attention(q, k[t], mask[t])
            x_t = att.unsqueeze(-2) @ n[t]
            x_t = x_t.squeeze(-2)
            x_t = torch.cat((x_t, s[t]), -1)
            h.append(x_t)
        h = torch.stack(h, dim=0)
        x = h[-1]
        if y is None: return x
        mask_t = mask[L1:L1 + L2].unsqueeze(-1)
        n_t = n[L1:L1 + L2]
        n_t = (mask_t * n_t).sum(-2)
        s_t = s[L1:L2 + L2]
        x_t = torch.cat((n_t, s_t), -1)
        x_t = torch.flip(x_t, (0,))
        b = x_t
        return x, b

    def forward(self, *args, **kwargs):
        args = iter(args)
        x = kwargs["x"] if "x" in kwargs else next(args)
        neighbor = kwargs["neighbor"] if "neighbor" in kwargs else next(args)
        try:
            n_predictions = kwargs["n_predictions"] if "n_predictions" in kwargs else next(args)
        except:
            n_predictions = 0

        stochastic = n_predictions > 0
        if neighbor is None:
            neighbor_shape = [_ for _ in x.shape]
            neighbor_shape.insert(-1, 0)
            neighbor = torch.empty(neighbor_shape, dtype=x.dtype, device=x.device)
        C = x.dim()
        if C < 3:
            x = x.unsqueeze(1)
            neighbor = neighbor.unsqueeze(1)
            if "y" in kwargs: kwargs["y"] = kwargs["y"].unsqueeze(1)
        N = x.size(1)

        neighbor = neighbor[:x.size(0)]
        h = self.enc(x, neighbor, y=kwargs.get("y", None))

        D = []
        for t in range(self.horizon):
            p_z = self.p_z(h)
            if stochastic:
                z = p_z.sample()
            else:
                z = p_z.mean
            d = self.dec(z, h)
            D.append(d)
            if t == self.horizon - 1: break
            zd = self.embed_zd(z, d)
            h = zd

        d = torch.stack(D)
        pred = torch.cumsum(d, 0)
        if stochastic:
            pred = pred.view(pred.size(0), n_predictions, -1, pred.size(-1)).permute(1, 0, 2, 3)
        pred = pred + x[-1, ..., :2]
        if C < 3: pred = pred.squeeze(1)
        return pred

    def learn(self, x, y, neighbor):
        h, b = self.enc(x, neighbor, y=y)
        D = []
        for t in range(self.horizon):
            p_z = self.p_z(h)
            q_z = self.q_z(h, b[t])
            z = q_z.rsample()
            d = self.dec(z, h)
            D.append(d)
            zd = self.embed_zd(z, d)
            h = zd

        d = torch.stack(D)
        pred = torch.cumsum(d, 0)
        pred = pred + x[-1, ..., :2]
        return pred
    def loss(self, err, kl):
        rec = err.mean()
        kl = kl.mean()
        return {
            "loss": kl+rec,
            "rec": rec,
            "kl": kl
        }
