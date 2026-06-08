"""Tests for the supervised bootstrap pipeline (agents/az/bootstrap.py).

Run with:
    . .venv/bin/activate && pytest tests/test_bootstrap.py -q
"""
import subprocess
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MM_SPEC = {"engine": "minimax", "params": {"time_budget": 0.02}}


def _fast_examples(n_games=1, seed=42):
    """Generate examples from n_games fast minimax-vs-minimax games."""
    from agents.az.bootstrap import teacher_game_examples
    examples = []
    for i in range(n_games):
        examples.extend(
            teacher_game_examples(_MM_SPEC, _MM_SPEC, seed=seed + i,
                                  temp_moves=6, max_plies=60)
        )
    return examples


# ---------------------------------------------------------------------------
# Test 1: basic shape and value constraints
# ---------------------------------------------------------------------------

class TestExampleFormat:
    """teacher_game_examples returns well-formed (planes, pi, z) tuples."""

    def setup_method(self):
        self.examples = _fast_examples(n_games=1, seed=7)

    def test_non_empty(self):
        assert len(self.examples) > 0, "Expected at least one training example"

    def test_planes_shape(self):
        for planes, pi, z in self.examples:
            assert planes.shape == (6, 9, 9), (
                f"planes shape {planes.shape} != (6, 9, 9)"
            )

    def test_pi_shape(self):
        for planes, pi, z in self.examples:
            assert pi.shape == (140,), f"pi shape {pi.shape} != (140,)"

    def test_pi_sums_to_one(self):
        for planes, pi, z in self.examples:
            assert abs(pi.sum() - 1.0) < 1e-5, (
                f"pi sums to {pi.sum():.6f}, expected ~1.0"
            )

    def test_pi_nonneg(self):
        for planes, pi, z in self.examples:
            assert (pi >= 0).all(), "pi contains negative values"

    def test_z_valid(self):
        for planes, pi, z in self.examples:
            assert z in {-1.0, 0.0, 1.0}, f"z={z!r} not in {{-1.0, 0.0, 1.0}}"


# ---------------------------------------------------------------------------
# Test 2: policy mass on legal actions
# ---------------------------------------------------------------------------

class TestPolicyOnLegalActions:
    """Policy target should not assign mass outside legal action indices."""

    def test_pi_mass_on_legal_actions(self):
        from agents.az.bootstrap import teacher_game_examples
        from agents.az.encoding import legal_action_mask

        # Get examples with state context: re-run a game manually.
        from core.state import initial_state
        from core.rules import apply_move, is_terminal, legal_moves
        from agents.minimax_agent import MinimaxAgent
        from agents.az.bootstrap import _candidates_to_pi
        from agents.az.encoding import encode_planes

        rng_seed = 13
        agent = MinimaxAgent(time_budget=0.02, seed=rng_seed)
        state = initial_state()
        ply = 0
        # Find the first position where we can verify mask alignment.
        found = False
        while not is_terminal(state) and ply < 20:
            analysis = agent.analyze(state)
            mask = legal_action_mask(state)
            pi = _candidates_to_pi(analysis.candidates, analysis.best_move, state)

            # The argmax of pi should be a legal action.
            best_idx = int(np.argmax(pi))
            assert mask[best_idx] == 1.0, (
                f"pi argmax action {best_idx} is not legal at ply {ply}"
            )

            # All mass should sit on legal action indices.
            illegal_mass = pi[mask == 0].sum()
            assert illegal_mass < 1e-5, (
                f"pi has {illegal_mass:.6f} mass on illegal actions at ply {ply}"
            )
            found = True

            state = apply_move(state, analysis.best_move)
            ply += 1

        assert found, "Did not find any position to check"


# ---------------------------------------------------------------------------
# Test 3: tiny end-to-end training check
# ---------------------------------------------------------------------------

class TestEndToEndTraining:
    """Generate a tiny dataset and verify a small net trains (loss decreases)."""

    def test_loss_finite_and_decreasing(self):
        import torch
        from agents.az.model import QuoridorNet
        from agents.az.train import examples_to_batch, train_step

        examples = _fast_examples(n_games=2, seed=99)
        assert len(examples) >= 4, "Need at least a few examples to train"

        net = QuoridorNet(channels=8, blocks=1)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        # Run multiple train steps; track first and last loss.
        import random
        rng = random.Random(0)
        indices = list(range(len(examples)))

        losses = []
        for _ in range(8):
            rng.shuffle(indices)
            mini = [examples[i] for i in indices[:min(32, len(indices))]]
            batch = examples_to_batch(mini)
            loss = train_step(net, optimizer, batch)
            losses.append(loss)

        # All losses should be finite.
        for i, loss in enumerate(losses):
            assert np.isfinite(loss), f"Loss at step {i} is not finite: {loss}"

        # The final loss should be lower than the first (net is learning).
        assert losses[-1] < losses[0], (
            f"Final loss {losses[-1]:.4f} >= first loss {losses[0]:.4f}; "
            "net is not learning on this small dataset"
        )


# ---------------------------------------------------------------------------
# Test 4: bootstrap.py must NOT import torch
# ---------------------------------------------------------------------------

class TestNoTorchImport:
    """agents.az.bootstrap must not pull in torch (subprocess check)."""

    def test_bootstrap_no_torch_in_subprocess(self):
        """Import bootstrap.py in a fresh Python process; assert torch absent."""
        code = (
            "import sys\n"
            "# Remove any pre-loaded torch from the environment.\n"
            "for key in list(sys.modules.keys()):\n"
            "    if 'torch' in key:\n"
            "        del sys.modules[key]\n"
            "import agents.az.bootstrap\n"
            "assert 'torch' not in sys.modules, "
            "f'torch was imported! modules: {[k for k in sys.modules if \"torch\" in k]}'\n"
            "print('OK: torch not imported')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            cwd=str(_repo_root()),
        )
        assert result.returncode == 0, (
            f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK: torch not imported" in result.stdout

    def test_bootstrap_source_no_torch_import(self):
        """bootstrap.py source code must not contain 'import torch'."""
        import os
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "az", "bootstrap.py",
        )
        with open(src_path) as f:
            source = f.read()
        assert "import torch" not in source, (
            "bootstrap.py source contains 'import torch' — remove it!"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
