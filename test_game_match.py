"""
test_game_match.py
==================
Correctness tests for game_match.py.

Primary sanity check: the pipeline recovers the true player/state/action
correspondence on a relabelled + per-player rescaled copy of the demo game,
and estimates c_hat within a few percent of the true scaling constants.

Also covers:
- load_game validation errors
- compute_keys values (normal + intrinsic modes)
- _compute_intrinsic_reward formula
- match with use_intrinsic=True (c_hat → 1.0 for pure relabelling)
"""

import pytest
import math

import game_match as gm


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_game_ab(true_scales=None):
    """Return (gameA, gameB, state_rename, player_rename, action_rename, true_scales).

    gameB is gameA relabelled (s→t, p→q, actions doubled) and rescaled.
    """
    if true_scales is None:
        true_scales = {"p1": 2.5, "p2": 0.4}

    raw_A = gm._make_demo_game()
    gameA = gm.load_game(raw_A)

    state_rename = {
        "s0": "t0", "s1": "t1", "s2": "t2",
        "s3": "t3", "s4": "t4", "s5": "t5", "s6": "t6",
    }
    player_rename = {"p1": "q1", "p2": "q2"}
    action_rename = {
        "L": "LL", "R": "RR",
        "A": "AA", "B": "BB", "X": "XX", "Y": "YY",
        "U": "UU", "V": "VV", "W": "WW",
        "C": "CC", "Z": "ZZ",
    }

    raw_B = gm._relabel_and_rescale(
        raw_A, state_rename, player_rename, action_rename, true_scales
    )
    gameB = gm.load_game(raw_B)
    return gameA, gameB, state_rename, player_rename, action_rename, true_scales


# ─────────────────────────────────────────────────────────────────────────────
# 1.  load_game validation
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadGame:
    def test_valid_game_loads(self):
        raw = gm._make_demo_game()
        g = gm.load_game(raw)
        assert g["root"] == "s0"
        assert set(g["players"]) == {"p1", "p2"}

    def test_missing_players_raises(self):
        raw = gm._make_demo_game()
        del raw["players"]
        with pytest.raises(ValueError, match="players"):
            gm.load_game(raw)

    def test_missing_root_raises(self):
        raw = gm._make_demo_game()
        del raw["root"]
        with pytest.raises(ValueError, match="root"):
            gm.load_game(raw)

    def test_root_not_in_states_raises(self):
        raw = gm._make_demo_game()
        raw["root"] = "nonexistent"
        with pytest.raises(ValueError, match="not found"):
            gm.load_game(raw)

    def test_terminal_missing_payoffs_raises(self):
        raw = gm._make_demo_game()
        del raw["states"]["s4"]["payoffs"]
        with pytest.raises(ValueError, match="payoffs"):
            gm.load_game(raw)

    def test_nonterminal_missing_transitions_raises(self):
        raw = gm._make_demo_game()
        del raw["states"]["s0"]["transitions"]
        with pytest.raises(ValueError, match="transitions"):
            gm.load_game(raw)

    def test_duplicate_profile_raises(self):
        raw = gm._make_demo_game()
        # Duplicate first profile in s0
        raw["states"]["s0"]["transitions"].append(
            raw["states"]["s0"]["transitions"][0]
        )
        with pytest.raises(ValueError, match="duplicate|transition"):
            gm.load_game(raw)

    def test_distribution_not_sum_to_1_raises(self):
        raw = gm._make_demo_game()
        raw["states"]["s0"]["transitions"][0]["next"]["s1"] = 0.9  # sums to 1.2
        with pytest.raises(ValueError, match="sum"):
            gm.load_game(raw)

    def test_cycle_raises(self):
        raw = gm._make_demo_game()
        # Add a back-edge s4 → s0 to create a cycle
        raw["states"]["s4"] = {
            "terminal": False,
            "actions": {"p1": ["X"]},
            "transitions": [
                {"profile": {"p1": "X"}, "next": {"s0": 1.0}},
            ],
        }
        with pytest.raises(ValueError, match="cycle|acyclic"):
            gm.load_game(raw)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  compute_keys (normal mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeKeys:
    def setup_method(self):
        raw = gm._make_demo_game()
        self.game = gm.load_game(raw)
        self.keys = gm.compute_keys(self.game)

    def test_terminal_values_equal_payoffs(self):
        V = self.keys["V"]
        states = self.game["states"]
        for s, sd in states.items():
            if sd.get("terminal", False):
                for p, v in sd["payoffs"].items():
                    assert V[s][p] == pytest.approx(v, abs=1e-9)

    def test_nonterminal_v_geq_terminal_values(self):
        # V[s0][p1] should be >= min terminal payoff for p1 (value ≥ -1)
        V = self.keys["V"]
        assert V["s0"]["p1"] >= -1.0 - 1e-9

    def test_root_inflow_is_zero(self):
        assert self.keys["inflow"]["s0"] == pytest.approx(0.0, abs=1e-9)

    def test_terminal_depth_positive(self):
        depth = self.keys["depth"]
        # All terminal states (s4, s5, s6) must be deeper than root (depth > 0)
        for s in ["s4", "s5", "s6"]:
            assert depth[s] >= 1

    def test_root_depth_zero(self):
        assert self.keys["depth"]["s0"] == 0

    def test_q_values_present_for_active_players(self):
        Q = self.keys["Q"]
        # p1 active at s0 → Q[s0][p1] should exist
        assert "p1" in Q["s0"]
        assert set(Q["s0"]["p1"].keys()) == {"L", "R"}

    def test_q_absent_for_nonactive_players(self):
        Q = self.keys["Q"]
        # p1 not active at s2 → Q[s2][p1] should be empty dict
        assert Q["s2"].get("p1", {}) == {}

    def test_v_equals_max_q_for_active(self):
        V = self.keys["V"]
        Q = self.keys["Q"]
        for s, q_players in Q.items():
            for p, q_actions in q_players.items():
                if q_actions:
                    assert V[s][p] == pytest.approx(max(q_actions.values()), abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  _compute_intrinsic_reward formula
# ─────────────────────────────────────────────────────────────────────────────

class TestIntrinsicReward:
    def setup_method(self):
        raw = gm._make_demo_game()
        self.game = gm.load_game(raw)

    def _lookup_for(self, state_name):
        sd = self.game["states"][state_name]
        active = sorted(sd["actions"])
        lookup = {
            tuple(t["profile"][p] for p in active): t
            for t in sd["transitions"]
        }
        return sd, lookup, active

    def test_nonactive_player_power_is_zero(self):
        # p1 is not active at s2
        sd, lookup, active = self._lookup_for("s2")
        R = gm._compute_intrinsic_reward(self.game["players"], sd, lookup, active)
        assert R["p1"] == pytest.approx(0.0, abs=1e-12)

    def test_s2_p2_power(self):
        # s2: only p2 active; transitions U→(s4:0.8,s5:0.2), V→(s5:0.5,s6:0.5), W→(s6:1.0)
        # R_p2(s2) = max(0.8²,0,0) + max(0.2²,0.5²,0) + max(0,0.5²,1.0²)
        #           = 0.64        + 0.25              + 1.0   = 1.89
        sd, lookup, active = self._lookup_for("s2")
        R = gm._compute_intrinsic_reward(self.game["players"], sd, lookup, active)
        assert R["p2"] == pytest.approx(1.89, abs=1e-9)

    def test_single_action_deterministic(self):
        # s3: p1 has {C}, p2 has {Z}; only transition is (C,Z) → s6: 1.0
        # For p1: others=[p2], combos=[("Z",)], n_combos=1
        #   s6: mean_prob(C) = 1.0 → best_sq = 1.0
        # R_p1(s3) = 1.0
        sd, lookup, active = self._lookup_for("s3")
        R = gm._compute_intrinsic_reward(self.game["players"], sd, lookup, active)
        assert R["p1"] == pytest.approx(1.0, abs=1e-9)

    def test_intrinsic_terminal_v_is_zero(self):
        keys_intr = gm.compute_keys(self.game, use_intrinsic=True)
        V = keys_intr["V"]
        for s, sd in self.game["states"].items():
            if sd.get("terminal", False):
                for p in self.game["players"]:
                    assert V[s][p] == pytest.approx(0.0, abs=1e-12)

    def test_intrinsic_nonterminal_v_reflects_power(self):
        # s2 has p2 with R=1.89; terminals are 0, so V[s2][p2] >= 1.89
        keys_intr = gm.compute_keys(self.game, use_intrinsic=True)
        assert keys_intr["V"]["s2"]["p2"] >= 1.89 - 1e-9

    def test_intrinsic_inactive_player_v_zero_when_no_future_power(self):
        # p1 is not active at s2, has no active states downstream → V[s2][p1]=0
        keys_intr = gm.compute_keys(self.game, use_intrinsic=True)
        assert keys_intr["V"]["s2"]["p1"] == pytest.approx(0.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Self-match correctness test  (primary sanity check)
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfMatch:
    """
    Game B is a relabelled + per-player rescaled copy of game A.
    The pipeline must recover:
      * the true player correspondence
      * the true state correspondence (all 7 states)
      * the true action correspondence (all action maps)
      * c_hat within a few percent of the true scaling constants
    """

    def setup_method(self):
        (self.gameA, self.gameB,
         self.state_rename, self.player_rename, self.action_rename,
         self.true_scales) = make_game_ab()

        self.result = gm.match(
            self.gameA, self.gameB,
            tau=0.4, sa_iters=20_000, seed=42,
        )

    # ── player map ──────────────────────────────────────────────────────────

    def test_player_map_correct(self):
        pm = self.result["player_map"]
        for pA, pB in self.player_rename.items():
            assert pm.get(pA) == pB, f"Expected {pA}→{pB}, got {pm}"

    # ── state map ───────────────────────────────────────────────────────────

    def test_state_map_correct(self):
        sm = self.result["state_map"]
        for sA, sB in self.state_rename.items():
            assert sm.get(sA) == sB, f"Expected {sA}→{sB}, got {sm}"

    def test_state_map_injective(self):
        sm = self.result["state_map"]
        assert len(sm.values()) == len(set(sm.values()))

    # ── action maps ─────────────────────────────────────────────────────────

    def test_action_maps_correct(self):
        am = self.result["action_maps"]
        for (sA, pA), amap in am.items():
            for aA, aB in amap.items():
                expected = self.action_rename[aA]
                assert aB == expected, (
                    f"State {sA}, player {pA}: expected {aA}→{expected}, got {aB}"
                )

    # ── c_hat accuracy ──────────────────────────────────────────────────────

    def test_c_hat_p1_within_5_percent(self):
        c = self.result["c_hat"]["p1"]
        true_c = self.true_scales["p1"]
        assert abs(c - true_c) / true_c < 0.05, (
            f"c_hat[p1]={c:.4f}, true={true_c}"
        )

    def test_c_hat_p2_within_5_percent(self):
        c = self.result["c_hat"]["p2"]
        true_c = self.true_scales["p2"]
        assert abs(c - true_c) / true_c < 0.05, (
            f"c_hat[p2]={c:.4f}, true={true_c}"
        )

    def test_c_hat_spread_near_zero(self):
        for p, spread in self.result["c_hat_spread"].items():
            assert spread < 0.05, f"c_hat_spread[{p}]={spread:.4f} too large"

    # ── score ────────────────────────────────────────────────────────────────

    def test_score_positive(self):
        assert self.result["score"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Intrinsic mode self-match
# ─────────────────────────────────────────────────────────────────────────────

class TestIntrinsicSelfMatch:
    """With intrinsic rewards (payoff-scale-free), c_hat must converge to 1.0."""

    def setup_method(self):
        (self.gameA, self.gameB,
         self.state_rename, self.player_rename, _,
         _) = make_game_ab(true_scales={"p1": 3.0, "p2": 0.2})

        self.result = gm.match(
            self.gameA, self.gameB,
            tau=0.4, sa_iters=20_000, seed=42,
            use_intrinsic=True,
        )

    def test_player_map_correct(self):
        pm = self.result["player_map"]
        for pA, pB in self.player_rename.items():
            assert pm.get(pA) == pB

    def test_state_map_correct(self):
        sm = self.result["state_map"]
        for sA, sB in self.state_rename.items():
            assert sm.get(sA) == sB, f"Expected {sA}→{sB}, got {sm}"

    def test_c_hat_near_one_for_both_players(self):
        # Intrinsic rewards ignore payoff scales → c_hat should be ~1.0
        for p, c in self.result["c_hat"].items():
            assert abs(c - 1.0) < 0.05, f"c_hat[{p}]={c:.4f}, expected ≈1.0"
