import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model


class AdvRelRotatEDB15K(Model):

    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_emb=None,
        text_emb=None,
        numeric_emb=None,
    ):

        super(AdvRelRotatEDB15K, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
        self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False,
        )

        self.img_dim = 256
        self.text_dim = 256

        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img_emb = img_pool(img_emb.view(-1, 64, 64))
        img_emb = img_emb.view(img_emb.size(0), -1)

        txt_pool = nn.AdaptiveAvgPool2d(output_size=(4, 64))
        text_emb = txt_pool(text_emb.view(-1, 12, 64))
        text_emb = text_emb.view(text_emb.size(0), -1)

        num_pool = nn.AdaptiveAvgPool2d(output_size=(4, 64))
        numeric_emb = txt_pool(numeric_emb.view(-1, 12, 64))
        numeric_emb = numeric_emb.view(numeric_emb.size(0), -1)

        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(
            False
        )
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(
            True
        )
        self.numeric_embeddings = nn.Embedding.from_pretrained(
            numeric_emb
        ).requires_grad_(False)

        # Old setting: 1-layer fc
        # self.img_proj = nn.Linear(self.img_dim, self.dim_e)
        # self.text_proj = nn.Linear(self.text_dim, self.dim_e)
        # self.numeric_proj = nn.Linear(self.text_dim, self.dim_e)

        # New setting: 2-layer mlp
        self.img_proj = nn.Sequential(
            nn.Linear(self.img_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e),
        )
        self.numeric_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e),
        )

        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item(),
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False,
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item(),
        )
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

        self.rel_gate = nn.Embedding(self.rel_tot, 1)
        nn.init.uniform_(
            tensor=self.rel_gate.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item(),
        )

    def gated_fusion(self, emb, rel):
        # emb: batch_size x dim
        # rel: batch_size x dim
        w = torch.sigmoid(emb * rel)
        return w * emb + (1 - w) * rel

    def get_joint_embeddings(self, es, ei, et, ea, rg):
        e = torch.stack((es, ei, et, ea), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores / torch.sigmoid(rg), dim=-1)
        context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
        return context_vectors

    def cal_score(self, embs):
        return self._calc(embs[0], embs[2], embs[1], "")

    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(
            1, 0, 2
        )
        re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(
            1, 0, 2
        )
        im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(
            1, 0, 2
        )
        im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(
            1, 0, 2
        )
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]
        ).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]
        ).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()

    def forward(self, data):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        mode = data["mode"]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        h_numeric_emb = self.numeric_proj(self.numeric_embeddings(batch_h))
        t_numeric_emb = self.numeric_proj(self.numeric_embeddings(batch_t))
        rg = self.rel_gate(batch_r)
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, h_numeric_emb, rg)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, t_numeric_emb, rg)
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score

    def forward_and_return_embs(self, data):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        mode = data["mode"]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        h_numeric_emb = self.numeric_proj(self.numeric_embeddings(batch_h))
        t_numeric_emb = self.numeric_proj(self.numeric_embeddings(batch_t))
        rg = self.rel_gate(batch_r)
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, h_numeric_emb, rg)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, t_numeric_emb, rg)
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score, [h_joint, r, t_joint]

    def get_batch_ent_embs(self, data):
        return self.ent_embeddings(data)

    def get_batch_vis_embs(self, data):
        return self.img_proj(self.img_embeddings(data))

    def get_batch_text_embs(self, data):
        return self.text_proj(self.text_embeddings(data))

    def get_batch_ent_multimodal_embs(self, data):
        return (
            self.ent_embeddings(data),
            self.img_proj(self.img_embeddings(data)),
            self.text_proj(self.text_embeddings(data)),
            self.numeric_proj(self.numeric_embeddings(data)),
        )

    def get_fake_score(
        self,
        batch_h,
        batch_r,
        batch_t,
        mode,
        fake_hi=None,
        fake_ti=None,
        fake_ht=None,
        fake_tt=None,
        fake_ha=None,
        fake_ta=None,
    ):
        if fake_hi is None or fake_ti is None or fake_ht is None or fake_tt is None:
            raise NotImplementedError
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        h_numeric_emb = self.numeric_proj(self.numeric_embeddings(batch_h))
        t_numeric_emb = self.numeric_proj(self.numeric_embeddings(batch_t))
        # the fake joint embedding
        rg = self.rel_gate(batch_r)
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, h_numeric_emb, rg)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, t_numeric_emb, rg)
        h_fake = self.get_joint_embeddings(h, fake_hi, fake_ht, fake_ha, rg)
        t_fake = self.get_joint_embeddings(t, fake_ti, fake_tt, fake_ta, rg)
        score_h = self.margin - self._calc(h_fake, t_joint, r, mode)
        score_t = self.margin - self._calc(h_joint, t_fake, r, mode)
        score_all = self.margin - self._calc(h_fake, t_fake, r, mode)
        return [score_h, score_t, score_all], [h_fake, r, t_fake]

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data["batch_h"]
        batch_t = data["batch_t"]
        batch_r = data["batch_r"]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h**2) + torch.mean(t**2) + torch.mean(r**2)) / 3
        return regul

    def get_attention_weight(self, h, t):
        h = torch.LongTensor([h])
        t = torch.LongTensor([t])
        h_s = self.ent_embeddings(h)
        t_s = self.ent_embeddings(t)
        h_img_emb = self.img_proj(self.img_embeddings(h))
        t_img_emb = self.img_proj(self.img_embeddings(t))
        h_text_emb = self.text_proj(self.text_embeddings(h))
        t_text_emb = self.text_proj(self.text_embeddings(t))
        # the fake joint embedding
        h_attn = self.get_attention(h_s, h_img_emb, h_text_emb)
        t_attn = self.get_attention(t_s, t_img_emb, t_text_emb)
        return h_attn, t_attn

    def attention_weight(self, es, ei, et, ea, rg):
        e = torch.stack((es, ei, et, ea), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores / torch.sigmoid(rg), dim=-1)
        return attention_weights

    def get_attention_weights(self, batch_h, batch_r):
        batch_h = torch.LongTensor(batch_h)
        batch_r = torch.LongTensor(batch_r)
        h = self.ent_embeddings(batch_h)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        h_numeric_emb = self.numeric_proj(self.numeric_embeddings(batch_h))
        rg = self.rel_gate(batch_r)
        weights_h = self.attention_weight(h, h_img_emb, h_text_emb, h_numeric_emb, rg)
        return weights_h
