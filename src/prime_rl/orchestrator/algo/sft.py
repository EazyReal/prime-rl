from __future__ import annotations

from prime_rl.orchestrator.algo.base import Algorithm


class SFTAlgorithm(Algorithm):
    """Supervised fine-tuning: cross-entropy on the source's target tokens.

    Assigns no credit — the target tokens themselves are the supervision, so
    the algorithm only routes action tokens to the ``ce`` loss and overrides no
    scoring hook. Where the targets come from (a frozen teacher model's fresh
    rollouts, or a static dataset's stored traces) is the Sampler's concern."""

    action_loss_type = "ce"
