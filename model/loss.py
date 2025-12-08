import torch

class LossComputer:
    """
    Pure Group-DRO version (without btl / without CVaR), with simple debug prints.

    is_robust = False: falls back to ERM (per_sample_losses.mean())
    is_robust = True : uses historical EMA-based Group-DRO
    """

    def __init__(
        self,
        is_robust,
        group_stats,
        gamma=0.1,
        adj=None,
        step_size=0.01,
        normalize_loss=False,
        device="cpu",
    ):
        self.device = device
        self.is_robust = is_robust
        self.gamma = gamma
        self.step_size = step_size
        self.normalize_loss = normalize_loss

        # Group-related statistics
        self.n_groups = group_stats["n_groups"]
        self.group_counts = group_stats["group_counts"].float().to(device)
        self.group_frac = group_stats["group_frac"].float().to(device)

        # Generalization adjustment
        self.adj = torch.zeros(self.n_groups, device=device) if adj is None \
            else torch.tensor(adj, dtype=torch.float32, device=device)

        # Adversarial distribution initialized as uniform
        self.adv_probs = torch.ones(self.n_groups, device=device) / self.n_groups

        # EMA of historical group losses
        self.exp_avg_loss = torch.zeros(self.n_groups, device=device)
        self.exp_avg_initialized = torch.zeros(self.n_groups, dtype=torch.bool, device=device)

        # Statistics & batch count
        self.reset_stats()

    def loss(self, per_sample_losses, group_idx=None, is_training=True):
        """
        per_sample_losses: [B]
        group_idx: [B], group id for each sample
        is_training: if True, update EMA and statistics; if False, only forward computation
        """
        if group_idx is None:
            return per_sample_losses.mean()

        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        if self.is_robust:
            actual_loss, weights = self.compute_robust_loss(group_loss)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        if is_training:
            self.update_exp_avg_loss(group_loss, group_count)
            self.update_stats(actual_loss.detach(), group_loss.detach(), group_count, weights)

        return actual_loss


    def compute_robust_loss(self, group_loss):
        """
        Standard Group-DRO:
        - Use historical EMA of group_loss + adjustment term as reference ref
        - Update adv_probs via exponentiated-gradient
        - Current batch robust_loss = group_loss · adv_probs
        """
        # Historical EMA + generalization adjustment
        ref = self.exp_avg_loss.detach().clone() + self._safe_adj_over_counts()

        if self.normalize_loss:
            # Normalize using z-score to avoid flattening differences by simple division by sum
            mean = ref.mean()
            std = ref.std(unbiased=False).clamp_min(1e-6)
            ref = (ref - mean) / std

        with torch.no_grad():
            w = torch.exp(self.step_size * ref)
            self.adv_probs = w / w.sum().clamp_min(1e-12)

        robust_loss = (group_loss * self.adv_probs).sum()
        return robust_loss, self.adv_probs.clone()


    def compute_group_avg(self, losses, group_idx):
        """
        Compute average loss per group based on group_idx
        """
        losses = losses.view(-1)
        group_idx = group_idx.view(-1).to(self.device)
        ids = torch.arange(self.n_groups, device=self.device)[:, None]  # [G,1]
        group_map = (group_idx == ids).float()  # [G,B]
        count = group_map.sum(1)                # [G]
        denom = count + (count == 0).float()
        return (group_map @ losses) / denom, count


    def update_exp_avg_loss(self, group_loss, group_count):
        """
        Update EMA only for groups that appear in this batch
        """
        mask = group_count > 0
        if mask.any():
            self.exp_avg_loss = torch.where(
                mask,
                self.gamma * self.exp_avg_loss + (1 - self.gamma) * group_loss,
                self.exp_avg_loss,
            )

    def update_stats(self, actual_loss, group_loss, group_count, weights=None):
        """
        Record some statistical information (optional, whether used or not)
        """
        # Sample-weighted avg_group_loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev = self.processed_data_counts / denom
        curr = group_count / denom

        self.avg_group_loss = prev * self.avg_group_loss + curr * group_loss

        # Batch-wise avg_actual_loss
        denom_b = self.batch_count + 1
        self.avg_actual_loss = (
            self.batch_count / denom_b * self.avg_actual_loss +
            1 / denom_b * actual_loss
        )

        # Sample counting
        self.processed_data_counts += group_count

        if self.is_robust and weights is not None:
            self.update_data_counts += group_count * (weights > 0).float()
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()

        self.batch_count += 1

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_batch_counts = torch.zeros(self.n_groups, device=self.device)
        self.avg_group_loss = torch.zeros(self.n_groups, device=self.device)
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.batch_count = 0


    def _safe_adj_over_counts(self):
        """
        Safe version of adj / sqrt(group_counts)
        """
        return self.adj / torch.sqrt(self.group_counts.clamp_min(1.0))
