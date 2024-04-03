import torch
import torch.nn.functional as F
import numpy as np
import platform


def dis_2_score(dis, return_numpy=True):
    """
    convert distance to score
    :param dis: distance
    :return: score
    """
    w = torch.linspace(1, 10, 10).to(dis.device)
    w_batch = w.repeat(dis.shape[0], 1)
    score = (dis * w_batch).sum(dim=1)
    if return_numpy:
        return score.cpu().numpy()
    else:
        return score

def base_emd_loss(x, y_true, dist_r=2):
    cdf_x = torch.cumsum(x, dim=-1)
    cdf_ytrue = torch.cumsum(y_true, dim=-1)
    if dist_r == 2:
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
    else:
        samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
    return samplewise_emd


class emd_loss(torch.nn.Module):
    """
    Earth Mover Distance loss
    """

    def __init__(self, dist_r=2,
                 use_l1loss=True, l1loss_coef=0.0):
        super(emd_loss, self).__init__()
        self.dist_r = dist_r
        self.use_l1loss = use_l1loss
        self.l1loss_coef = l1loss_coef

    def check_type_forward(self, in_types):
        assert len(in_types) == 2

        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0
        assert x_type.ndim == 2
        assert y_type.ndim == 2

    def forward(self, x, y_true):
        self.check_type_forward((x, y_true))

        cdf_x = torch.cumsum(x, dim=-1)
        cdf_ytrue = torch.cumsum(y_true, dim=-1)
        if self.dist_r == 2:
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
        else:
            samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
        loss = torch.mean(samplewise_emd)
        if self.use_l1loss:
            rate_scale = torch.tensor([float(i + 1) for i in range(x.size()[1])], dtype=x.dtype, device=x.device)
            x_mean = torch.mean(x * rate_scale, dim=-1)
            y_true_mean = torch.mean(y_true * rate_scale, dim=-1)
            l1loss = torch.mean(torch.abs(x_mean - y_true_mean))
            loss += l1loss * self.l1loss_coef
        return loss


class MPEMDLoss(torch.nn.Module):
    def __init__(self, dist_r=2, eps=1e-6, beta=0.7, k=1.2):
        super(MPEMDLoss, self).__init__()
        self.dist_r = dist_r
        self.eps = eps
        self.beta = beta
        self.k = k
        self.emd = base_emd_loss
        # if system is linux, compile the emd loss
        if platform.system() == 'Linux':
            self.emd = torch.compile(base_emd_loss)

    def forward(self, x, y_true):
        patch_num = x.size(1)
        x_flatten = x.view(-1, x.size(-1))
        # copy y_true patch_num times at dim 1
        y_true_flatten = y_true.repeat(1, patch_num).view(-1, y_true.size(-1))
        loss = self.emd(x_flatten, y_true_flatten)
        loss = loss.contiguous().view(-1, patch_num)
        eps = torch.ones_like(loss) * self.eps
        emdc = torch.max(eps, 1 - self.k * loss)
        weight = 1 - torch.pow(emdc, self.beta)
        loss = torch.mean(loss * weight)
        return loss


class SupCon(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SupCon, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Calculate the supervised contrastive loss

        :param embeddings: Tensor of shape (batch_size, feature_dim), the feature embeddings.
        :param labels: Tensor of shape (batch_size,), the labels of the embeddings.
        :return: The supervised contrastive loss.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute the cosine similarity matrix
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1) / self.temperature

        # Mask for removing positive diagonal elements
        diag_mask = ~(torch.eye(batch_size, device=device).bool())

        # Exponential mask for the numerator
        exp_logits = torch.exp(cosine_sim) * diag_mask

        # Compute sum of exp logits
        log_prob = exp_logits / exp_logits.sum(dim=1, keepdim=True)

        # Create mask for positive samples
        labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Compute Supervised Contrastive Loss
        loss = -(labels_eq * diag_mask).float() * torch.log(log_prob + 1e-8)
        loss = loss.sum() / batch_size
        # loss = F.sigmoid(loss)

        return loss


class SupCRLoss(torch.nn.Module):
    def __init__(self, margin=1.0, temperature=0.07, m=None):
        """
        Initialize the Supervised Contrastive Regression Loss module.
        :param margin: Margin to define the boundary for dissimilar targets.
        :param temperature: Temperature scaling to control the separation of embeddings.
        """
        super(SupCRLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.m = m

    def forward(self, embeddings, targets):
        """
        Forward pass to compute the SupCR loss.
        :param embeddings: Tensor of shape (batch_size, embedding_dim), embedding representations of inputs.
        :param targets: Tensor of shape (batch_size,), continuous target values associated with each input.
        :return: The computed SupCR loss.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise cosine similarity
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        # Compute target differences matrix
        target_diffs = targets.unsqueeze(1) - targets.unsqueeze(0)

        # Apply margin to target differences
        target_diffs = torch.abs(target_diffs) - self.margin
        target_diffs = torch.clamp(target_diffs, min=0.0)

        # Calculate positive and negative masks
        positive_mask = target_diffs.eq(0).float() - torch.eye(target_diffs.shape[0], device=embeddings.device).float()
        negative_mask = target_diffs.gt(0).float()

        # Compute loss
        loss_positives = -torch.log(torch.exp(sim_matrix * positive_mask / self.temperature) + 1e-6).mean()
        loss_negatives = torch.log(torch.exp(sim_matrix * negative_mask / self.temperature) + 1e-6).mean()

        if self.m:
            loss = self.m - loss_positives + loss_negatives
        else:
            loss = loss_positives + loss_negatives

        return loss


if __name__ == "__main__":
    emd = MPEMDLoss()
    x = torch.rand(32, 5, 10)
    y = torch.rand(32, 5, 10)
    loss = emd(x, y)
