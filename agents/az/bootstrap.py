"""Supervised bootstrap data generation for AlphaZero.

Plays games between two teacher engines (minimax or mcts) and collects
(planes, pi, z) examples that can be used to train QuoridorNet via imitation.

IMPORTANT: This module must NOT pull in torch so that ProcessPoolExecutor workers
stay lightweight (numpy + core + agents engines + encoding only).
"""
import numpy as np
import random

from core.state import initial_state
from core.rules import apply_move, is_terminal, winner, legal_moves
from agents.az.encoding import encode_planes, move_to_action, N_ACTIONS
from agents.heuristics import WIN_SCORE

# Temperature used when building softmax over candidate scores.
_SCORE_TEMP = 1.0


def _make(spec, seed=None):
    """Construct an agent from a spec dict.

    spec: {"engine": "minimax"|"mcts", "params": {...}}
    seed is forwarded to the agent constructor.
    """
    engine = spec["engine"]
    params = dict(spec.get("params", {}))
    if seed is not None:
        params.setdefault("seed", seed)
    if engine == "minimax":
        from agents.minimax_agent import MinimaxAgent
        return MinimaxAgent(**params)
    elif engine == "mcts":
        from agents.mcts_agent import MCTSAgent
        return MCTSAgent(**params)
    else:
        raise ValueError(f"Unknown engine: {engine!r}. Use 'minimax' or 'mcts'.")


def _candidates_to_pi(candidates, best_move, state, score_temp=_SCORE_TEMP):
    """Build a soft policy target vector (length N_ACTIONS, float32) from a
    teacher's candidate list.

    Args:
        candidates: list of (move, score) from Analysis.candidates.
        best_move:  fallback move when candidates is empty.
        state:      current game state (used for canonical action encoding).
        score_temp: temperature divisor for scores before softmax.

    Returns:
        np.ndarray of shape (N_ACTIONS,), dtype float32, summing to 1.
    """
    pi = np.zeros(N_ACTIONS, dtype=np.float32)

    if not candidates:
        # Fall back to one-hot on best_move.
        idx = move_to_action(best_move, state)
        pi[idx] = 1.0
        return pi

    # Map candidate moves to canonical action indices and their scores.
    idxs = []
    scores = []
    for move, score in candidates:
        try:
            idx = move_to_action(move, state)
        except (KeyError, ValueError):
            continue  # skip any move that doesn't map cleanly
        idxs.append(idx)
        # Clamp WIN_SCORE-ish values to avoid overflow in exp.
        clamped = max(-WIN_SCORE, min(WIN_SCORE, score))
        scores.append(clamped / score_temp)

    if not idxs:
        # All candidates failed to map; fall back to one-hot.
        idx = move_to_action(best_move, state)
        pi[idx] = 1.0
        return pi

    scores_arr = np.array(scores, dtype=np.float64)
    # Numerically stable softmax: subtract max before exp.
    scores_arr -= scores_arr.max()
    exp_scores = np.exp(scores_arr)
    exp_scores /= exp_scores.sum()

    for i, idx in enumerate(idxs):
        pi[idx] += float(exp_scores[i])

    # Normalise (guard against floating-point residuals).
    total = pi.sum()
    if total > 0:
        pi /= total

    return pi


def teacher_game_examples(spec0, spec1, seed,
                          temp_moves=12, max_plies=200,
                          explore_eps=0.15):
    """Play one game between two teacher engines and collect training examples.

    Args:
        spec0:       Engine spec for player 0, e.g. {"engine":"minimax","params":{"time_budget":0.05}}.
        spec1:       Engine spec for player 1.
        seed:        Integer seed for the game's RNG (move sampling + exploration).
        temp_moves:  Number of early plies where stochastic move selection is used.
        max_plies:   Hard cap on game length.
        explore_eps: Probability of playing a uniformly random legal move instead
                     of sampling from the teacher's distribution (early phase only).

    Returns:
        list of (planes np.ndarray(6,9,9), pi np.ndarray(140,), z float)
    """
    rng = random.Random(seed)

    # Build agents with per-player sub-seeds so they are independent but
    # reproducible from the top-level seed.
    agent0 = _make(spec0, seed=rng.randrange(1 << 30))
    agent1 = _make(spec1, seed=rng.randrange(1 << 30))
    agents = [agent0, agent1]

    state = initial_state()
    history = []  # (planes, pi_vec, player)
    ply = 0

    while not is_terminal(state) and ply < max_plies:
        player = state.turn
        agent = agents[player]

        # Ask the teacher for analysis.
        analysis = agent.analyze(state)

        # Build the policy target from the teacher's candidates.
        pi_vec = _candidates_to_pi(
            analysis.candidates, analysis.best_move, state
        )

        # Record (planes, pi, player) BEFORE making the move.
        history.append((encode_planes(state), pi_vec, player))

        # Choose which move to actually play.
        if ply < temp_moves:
            # Exploration phase.
            if rng.random() < explore_eps:
                # Uniformly random legal move.
                legal = legal_moves(state)
                move = rng.choice(legal)
            else:
                # Sample from the teacher's candidate distribution.
                if analysis.candidates:
                    moves_list = [c[0] for c in analysis.candidates]
                    scores_raw = [c[1] for c in analysis.candidates]
                    # Build sampling weights via softmax.
                    s = np.array(scores_raw, dtype=np.float64)
                    s = np.clip(s, -WIN_SCORE, WIN_SCORE) / _SCORE_TEMP
                    s -= s.max()
                    w = np.exp(s)
                    w /= w.sum()
                    # numpy choice with weights.
                    move = moves_list[
                        int(np.random.default_rng(
                            rng.randrange(1 << 30)
                        ).choice(len(moves_list), p=w))
                    ]
                else:
                    move = analysis.best_move
        else:
            # Greedy phase: play the teacher's best move.
            move = analysis.best_move

        state = apply_move(state, move)
        ply += 1

    # Determine outcome.
    w = winner(state)  # None if ply cap reached

    examples = []
    for planes, pi_vec, player in history:
        if w is None:
            z = 0.0
        elif w == player:
            z = 1.0
        else:
            z = -1.0
        examples.append((planes, pi_vec, z))

    return examples


def worker(args):
    """Top-level function suitable for ProcessPoolExecutor.map.

    Args:
        args: tuple (spec0, spec1, seed, temp_moves, max_plies, explore_eps)

    Returns:
        list of (planes, pi, z) examples from one game.
    """
    spec0, spec1, seed, temp_moves, max_plies, explore_eps = args
    return teacher_game_examples(spec0, spec1, seed,
                                 temp_moves=temp_moves,
                                 max_plies=max_plies,
                                 explore_eps=explore_eps)
