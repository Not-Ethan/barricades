import os

from agents.base import Agent, Analysis
from agents.az.model import QuoridorNet, NetWrapper
from agents.az.mcts_nn import PUCTSearch

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "..",
                            "models", "az_smoke.pt")


class AZAgent(Agent):
    name = "az"

    def __init__(self, checkpoint=None, sims=120, c_puct=1.5,
                 channels=32, blocks=3, seed=None):
        net = QuoridorNet(channels=channels, blocks=blocks)
        path = checkpoint or DEFAULT_CKPT
        if os.path.exists(path):
            import torch
            try:
                net.load_state_dict(torch.load(path, map_location="cpu"))
            except Exception:
                pass    # shape mismatch / corrupt -> use fresh net
        self._wrap = NetWrapper(net)
        self._sims = sims
        self._c_puct = c_puct
        self._seed = seed

    def analyze(self, state):
        search = PUCTSearch(self._wrap, sims=self._sims, c_puct=self._c_puct,
                            seed=self._seed)
        move, _, info = search.run(state)
        cands = sorted(info["visits"].items(), key=lambda kv: kv[1], reverse=True)
        return Analysis(best_move=move, value=info["value"],
                        candidates=[(m, float(n)) for m, n in cands[:8]],
                        stats={"sims": info["sims"]})

    def select_move(self, state):
        return self.analyze(state).best_move
