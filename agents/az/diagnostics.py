"""Per-iteration self-play diagnostics for tracking the race-collapse mechanism.

The plateau symptom is game-length collapse (already logged by the campaign). These
diagnostics expose the *mechanism*: whether the policy prior collapses off wall
actions. Action encoding (see agents/az/train.py LR_PERM): indices 0..11 are pawn
moves; 12..139 are wall placements (12..75 H walls, 76..139 V walls).
"""
import numpy as np

WALL_ACTION_START = 12  # indices >= this are wall placements


def policy_diagnostics(examples):
    """Compute wall-usage + entropy diagnostics from raw self-play examples.

    examples: list of (planes(6,9,9), pi(140), z, feats). `pi` is the MCTS
    visit-count policy at the root. Computed on raw (pre-augmentation) examples.

    Returns dict:
      n                : number of examples
      wall_mass        : mean total policy mass on wall actions (prior-collapse signal)
      wall_argmax_rate : fraction of positions whose top move is a wall
      entropy          : mean Shannon entropy (nats) of the root policy
    """
    if not examples:
        return dict(n=0, wall_mass=0.0, wall_argmax_rate=0.0, entropy=0.0)
    pis = np.stack([np.asarray(e[1], dtype=np.float64) for e in examples])  # (N,140)
    sums = pis.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    pis = pis / sums
    wall_mass = float(pis[:, WALL_ACTION_START:].sum(axis=1).mean())
    wall_argmax_rate = float((pis.argmax(axis=1) >= WALL_ACTION_START).mean())
    entropy = float((-(pis * np.log(pis + 1e-12)).sum(axis=1)).mean())
    return dict(n=len(examples), wall_mass=round(wall_mass, 4),
                wall_argmax_rate=round(wall_argmax_rate, 4),
                entropy=round(entropy, 4))
