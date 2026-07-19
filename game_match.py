"""
game_match.py
=============
Approximate correspondence between two finite acyclic stochastic games.

The games are given as validated Python dicts (see ``load_game``).
Rewards in the two games may be rescaled by an unknown *positive* constant
per player; the pipeline is invariant to that.

Public API
----------
load_game(data)          – validate & load a game from a JSON-derived dict
compute_keys(game, ...)  – compute V, Q, inflow, depth via one DAG pass
match_players(...)       – match players on scale-invariant structural features
match_states(...)        – match states within successors of matched states
match_actions(...)       – match actions on Q-value similarity
build_f0(...)            – build the initial matching f0
score(f, ...)            – evaluate a full matching
local_search(...)        – simulated-annealing improvement of f0
match(gameA, gameB)      – end-to-end entry point

Intrinsic reward option
-----------------------
Pass ``use_intrinsic=True`` to ``compute_keys``, ``build_f0``, or ``match``
to replace real terminal payoffs with a scale-invariant *individual power*
metric.  For each non-terminal state s and player i the immediate reward is:

    R_i(s) = Σ_{s''} max_{b_i ∈ A_i(s)} ( E_{b_{-i}~unif(A_{-i}(s))} P(s''|s,(b_i,b_{-i})) )²

Intuitively: for each possible next state s'', player i picks the action that
maximises the *squared mean* transition probability to s'' (mean taken over
the other players' uniform mixture), then sums over all next states.
Terminal states contribute 0 (no further influence).  The metric is
invariant to payoff scaling, so c_hat converges to 1 on a pure relabelling.
"""

from __future__ import annotations

import json
import math
import random
from itertools import product as cartesian_product
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

# ─────────────────────────────────────────────────────────────────────────────
# Constants / configurable defaults
# ─────────────────────────────────────────────────────────────────────────────
EPS: float = 1e-9

# Similarity weights: v=value, i=inflow, d=depth, t=transition structure
DEFAULT_WEIGHTS: dict = {"v": 0.4, "i": 0.3, "d": 0.3, "t": 1.5}

# Drop matched pairs whose similarity falls below this threshold
DEFAULT_TAU: float = 0.5

# Simulated-annealing parameters
DEFAULT_SA_ITERS: int = 20_000
DEFAULT_SA_T0: float = 1.0
DEFAULT_SA_ALPHA: float = 0.9997   # geometric cooling; T drops ~1000× over 20k steps


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Loading / validation
# ═════════════════════════════════════════════════════════════════════════════

def load_game(data: dict) -> dict:
    """Validate and load a game from a JSON-derived dict.

    Checks
    ------
    * Required top-level keys (``players``, ``root``, ``states``) are present.
    * Root state exists.
    * Terminal states have ``payoffs``; non-terminal states have ``actions``
      and ``transitions``.
    * Every joint profile of active players appears **exactly once** in
      ``transitions``.
    * Each ``next`` distribution sums to 1 (tolerance ±1e-6).
    * The transition graph is a DAG (verified via networkx).

    Returns the dict augmented with ``_graph`` (a ``networkx.DiGraph``).
    """
    for key in ("players", "root", "states"):
        if key not in data:
            raise ValueError(f"Missing required top-level field '{key}'")

    players: List[str] = list(data["players"])
    root: str = data["root"]
    states: dict = data["states"]

    if root not in states:
        raise ValueError(f"Root '{root}' not found in states")

    G: nx.DiGraph = nx.DiGraph()
    G.add_nodes_from(states)

    for s, sd in states.items():
        terminal = bool(sd.get("terminal", False))
        if terminal:
            if "payoffs" not in sd:
                raise ValueError(f"Terminal state '{s}': missing 'payoffs'")
        else:
            for fld in ("actions", "transitions"):
                if fld not in sd:
                    raise ValueError(f"Non-terminal state '{s}': missing '{fld}'")

            active = sorted(sd["actions"])
            if not active:
                raise ValueError(f"State '{s}': non-terminal but has no active players")

            # Expected number of joint profiles = product of action counts
            expected = 1
            for p in active:
                expected *= len(sd["actions"][p])

            n_trans = len(sd["transitions"])
            if n_trans != expected:
                raise ValueError(
                    f"State '{s}': expected {expected} transition(s) "
                    f"(one per joint profile), found {n_trans}"
                )

            seen: set = set()
            for t in sd["transitions"]:
                key_tuple = tuple(t["profile"][p] for p in active)
                if key_tuple in seen:
                    raise ValueError(
                        f"State '{s}': duplicate joint profile "
                        f"{dict(zip(active, key_tuple))}"
                    )
                seen.add(key_tuple)

                total = sum(t["next"].values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"State '{s}', profile {key_tuple}: "
                        f"next-distribution sums to {total:.7g}, not 1"
                    )

                for s_next in t["next"]:
                    if s_next not in states:
                        raise ValueError(
                            f"State '{s}': successor '{s_next}' not in states"
                        )
                    G.add_edge(s, s_next)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Game graph contains a cycle (must be acyclic)")

    return {
        "players": players,
        "root": root,
        "states": states,
        "_graph": G,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Key computation  (one backward + one forward pass)
# ═════════════════════════════════════════════════════════════════════════════

def compute_keys(game: dict, use_intrinsic: bool = False) -> dict:
    """Compute V, Q, inflow, depth via a single backward + forward DAG pass.

    Reference policy: all active players other than the one whose Q we compute
    play **uniform-random** over their actions.

    Terminal s (use_intrinsic=False):
        V_i[s] = payoffs[s][i]

    Terminal s (use_intrinsic=True):
        V_i[s] = 0   (terminal states have no further influence)

    Non-terminal s, player i **active**:
        Q_i[s][a_i] = imm_i(s) + mean_{-i joint actions} Σ_{s'} P(s'|s,(a_i,−i)) V_i[s']
        V_i[s]      = max_{a_i} Q_i[s][a_i]

    Non-terminal s, player i **not active**:
        V_i[s] = imm_i(s) + mean_{all joint profiles} Σ_{s'} P(s'|s,profile) V_i[s']

    where imm_i(s) = R_i(s)  when use_intrinsic=True, else 0.
    See ``_compute_intrinsic_reward`` for the definition of R_i(s).

    inflow[s'] = Σ_s  mean_{profiles at s} P(s'|s,profile)   (root inflow = 0)
    depth[s]   = length of longest path root → s
    """
    players: List[str] = game["players"]
    root: str = game["root"]
    states: dict = game["states"]
    G: nx.DiGraph = game["_graph"]

    # Work only on states reachable from root
    reachable = nx.descendants(G, root) | {root}
    sub = G.subgraph(reachable).copy()
    topo: List[str] = list(nx.topological_sort(sub))   # root first

    # ── Backward pass (process states from leaves back to root) ───────────────
    V: Dict[str, Dict[str, float]] = {}
    Q: Dict[str, Dict[str, Dict[str, float]]] = {}

    for s in reversed(topo):
        sd = states[s]
        V[s] = {}
        Q[s] = {}

        if sd.get("terminal", False):
            for p in players:
                # With intrinsic rewards terminal states contribute 0
                V[s][p] = 0.0 if use_intrinsic else float(sd["payoffs"].get(p, 0.0))
            continue

        active = sorted(sd["actions"])
        # Fast profile lookup: tuple of actions (ordered by active) → transition
        lookup = {
            tuple(t["profile"][p] for p in active): t
            for t in sd["transitions"]
        }

        # Immediate intrinsic power reward at this state (0 when not used)
        R_intr = (
            _compute_intrinsic_reward(players, sd, lookup, active)
            if use_intrinsic
            else {p: 0.0 for p in players}
        )

        for i in players:
            imm = R_intr[i]   # 0.0 when use_intrinsic=False
            if i in sd["actions"]:
                # ── Active player: compute Q values then V = max Q ────────────
                others = [p for p in active if p != i]
                other_act_lists = [sd["actions"][p] for p in others]
                # All joint combos of other players' actions
                combos = list(cartesian_product(*other_act_lists)) if others else [()]
                Q[s][i] = {}
                for a_i in sd["actions"][i]:
                    q_sum = 0.0
                    for combo in combos:
                        # Build full joint profile including a_i
                        prof = dict(zip(others, combo))
                        prof[i] = a_i
                        key = tuple(prof[p] for p in active)
                        t = lookup[key]
                        q_sum += sum(
                            pr * V[s_next][i] for s_next, pr in t["next"].items()
                        )
                    # imm is the same for every a_i, so argmax is unchanged
                    Q[s][i][a_i] = imm + q_sum / len(combos)
                V[s][i] = max(Q[s][i].values())
            else:
                # ── Non-active player: average V over all joint profiles ───────
                # imm = R_intr[i] is the passing-action power for this player
                V[s][i] = imm + sum(
                    sum(pr * V[s_next][i] for s_next, pr in t["next"].items())
                    for t in sd["transitions"]
                ) / len(sd["transitions"])

    # ── Forward pass: inflow and depth ────────────────────────────────────────
    inflow: Dict[str, float] = {s: 0.0 for s in topo}
    depth: Dict[str, int] = {s: 0 for s in topo}

    for s in topo:
        sd = states[s]
        if not sd.get("terminal", False):
            n_prof = len(sd["transitions"])
            for t in sd["transitions"]:
                for s_next, pr in t["next"].items():
                    if s_next in inflow:
                        # Contribution = prob under uniform profile distribution
                        inflow[s_next] += pr / n_prof
        # Propagate depth to children (longest-path DP)
        for s_child in sub.successors(s):
            depth[s_child] = max(depth[s_child], depth[s] + 1)

    return {
        "topo_order": topo,
        "V": V,
        "Q": Q,
        "inflow": inflow,
        "depth": depth,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Intrinsic power reward
# ═════════════════════════════════════════════════════════════════════════════

def _compute_intrinsic_reward(
    players: List[str],
    sd: dict,
    lookup: dict,
    active: List[str],
) -> Dict[str, float]:
    """Compute the intrinsic power reward R_i(s) for one non-terminal state s.

    For active player i:

        R_i(s) = Σ_{s''} max_{b_i ∈ A_i(s)} ( E_{b_{-i}~unif(A_{-i}(s))} P(s''|s,(b_i,b_{-i})) )²

    Concretely for each reachable next state s'' and each candidate action b_i:
      1. Average P(s''|s,(b_i,b_{-i})) over all uniform combos of other players.
      2. Square that average (square is OUTSIDE the expectation).
      3. Take the max over b_i.
    Then sum over all s''.

    For non-active player i (no actions at s) the same formula is used with i's
    only action being "passing": i has no influence on the profile, so the max
    is over a single action and b_{-i} ranges over every active player's actions.

    Parameters
    ----------
    players : list of all player names
    sd      : state dict for s (non-terminal)
    lookup  : profile-tuple → transition dict (pre-built by compute_keys)
    active  : sorted list of active-player names at s
    """
    # Collect all reachable next-state names at this state
    all_s_next: set = set()
    for t in lookup.values():
        all_s_next.update(t["next"])

    R: Dict[str, float] = {}

    for i in players:
        if i in sd["actions"]:
            # Active player: choose among i's own actions.
            others = [p for p in active if p != i]
            i_actions: list = list(sd["actions"][i])
        else:
            # Non-active player: single "passing" action, no influence on the
            # profile; b_{-i} ranges over all active players' actions.
            others = list(active)
            i_actions = [None]

        other_act_lists = [sd["actions"][p] for p in others]
        combos = list(cartesian_product(*other_act_lists)) if others else [()]
        n_combos = len(combos)

        r_val = 0.0
        for s_next in all_s_next:
            best_sq = 0.0
            for a_i in i_actions:
                # Mean transition probability to s_next over uniform b_{-i}
                mean_prob = 0.0
                for combo in combos:
                    prof = dict(zip(others, combo))
                    if a_i is not None:
                        prof[i] = a_i
                    key = tuple(prof[p] for p in active)
                    mean_prob += lookup[key]["next"].get(s_next, 0.0)
                mean_prob /= n_combos
                # Square is outside the expectation
                best_sq = max(best_sq, mean_prob ** 2)
            r_val += best_sq

        R[i] = r_val

    return R


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Similarity helpers
# ═════════════════════════════════════════════════════════════════════════════

def _sim(x: np.ndarray, y: np.ndarray) -> float:
    """Normalised vector similarity in [0, 1].

    sim(x, y) = 1 − ‖x−y‖ / (‖x‖ + ‖y‖ + ε)

    Returns 1.0 when x == y (including both zero), 0 when the vectors point
    in opposite directions with equal magnitude.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = float(np.linalg.norm(x - y))
    denom = float(np.linalg.norm(x)) + float(np.linalg.norm(y)) + EPS
    return 1.0 - diff / denom


def _sim_scalar(a: float, b: float) -> float:
    """Scalar version of the normalised similarity."""
    return 1.0 - abs(a - b) / (abs(a) + abs(b) + EPS)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Player matching
# ═════════════════════════════════════════════════════════════════════════════

def match_players(
    gameA: dict,
    keysA: dict,
    gameB: dict,
    keysB: dict,
    tau: float = DEFAULT_TAU,
) -> dict:
    """Match players on scale-invariant structural features (Hungarian).

    Feature vector per player (z-normalised across both games' player sets):
      [#active_states,  total_#actions,  mean_actions_per_active_state,
       fraction_of_states_where_active]

    Returns ``{playerA: playerB}`` for pairs whose similarity ≥ ``tau``.
    """
    def _feats(player: str, states: dict) -> np.ndarray:
        n_total = len(states)
        active_sds = [
            sd for sd in states.values()
            if not sd.get("terminal", False) and player in sd.get("actions", {})
        ]
        n_act = len(active_sds)
        total_a = sum(len(sd["actions"][player]) for sd in active_sds)
        mean_a = total_a / max(n_act, 1)
        frac = n_act / max(n_total, 1)
        return np.array([n_act, total_a, mean_a, frac], dtype=float)

    pA: List[str] = gameA["players"]
    pB: List[str] = gameB["players"]
    fA = np.array([_feats(p, gameA["states"]) for p in pA])
    fB = np.array([_feats(p, gameB["states"]) for p in pB])

    # Z-normalise across the union of all players from both games
    all_f = np.vstack([fA, fB])
    mu = all_f.mean(axis=0)
    sigma = all_f.std(axis=0) + EPS
    fA_n = (fA - mu) / sigma
    fB_n = (fB - mu) / sigma

    nA, nB = len(pA), len(pB)
    sim_mat = np.array(
        [[_sim(fA_n[i], fB_n[j]) for j in range(nB)] for i in range(nA)]
    )

    rows, cols = linear_sum_assignment(-sim_mat)
    return {
        pA[r]: pB[c]
        for r, c in zip(rows, cols)
        if sim_mat[r, c] >= tau
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Per-player scale estimation
# ═════════════════════════════════════════════════════════════════════════════

def _bootstrap_c_hat(
    player_map: dict,
    keysA: dict,
    keysB: dict,
) -> dict:
    """Bootstrap scale estimate: c_hat_i = median|V^B_{π(i)}| / median|V^A_i|.

    Used before any state matching is available.
    """
    VA, VB = keysA["V"], keysB["V"]
    c_hat: dict = {}
    for pA, pB in player_map.items():
        vals_A = [abs(v[pA]) for v in VA.values() if abs(v[pA]) > EPS]
        vals_B = [abs(v[pB]) for v in VB.values() if abs(v[pB]) > EPS]
        scale_A = float(np.median(vals_A)) if vals_A else 1.0
        scale_B = float(np.median(vals_B)) if vals_B else 1.0
        c_hat[pA] = scale_B / max(scale_A, EPS)
    return c_hat


def _refine_c_hat(
    player_map: dict,
    state_map: dict,
    keysA: dict,
    keysB: dict,
    c_hat_prev: dict,
    eps: float = 1e-3,
) -> Tuple[dict, dict]:
    """Refine c_hat from the current matched state pairs.

    c_hat_i = median( V^B_{π(i)}[s'] / V^A_i[s] )
              over matched (s, s') with |V^A_i[s]| > eps

    Also returns a *spread* dict: relative std of the ratios per player.
    A large spread means the current state matching is wrong.
    Falls back to the previous estimate when no matched pairs are available.
    """
    VA, VB = keysA["V"], keysB["V"]
    c_hat = dict(c_hat_prev)
    spread: dict = {}
    for pA, pB in player_map.items():
        ratios = [
            VB[sB][pB] / VA[sA][pA]
            for sA, sB in state_map.items()
            if abs(VA[sA][pA]) > eps
        ]
        if ratios:
            c_hat[pA] = float(np.median(ratios))
            spread[pA] = float(np.std(ratios)) / (abs(c_hat[pA]) + EPS)
        else:
            spread[pA] = 0.0
    return c_hat, spread


# ═════════════════════════════════════════════════════════════════════════════
# 6.  State matching
# ═════════════════════════════════════════════════════════════════════════════

def _reach_probs(game: dict) -> Dict[str, float]:
    """Probability of visiting each state from the root under uniform policies.

    All active players play uniform-random, and each joint profile is weighted
    equally (as elsewhere in this module).  For a DAG this is a single forward
    pass in topological order:

        reach[root]   = 1
        reach[s_next] += reach[s] · mean_{profiles at s} P(s_next | s, profile)

    Used only to *order* processing of states in ``match_states`` so that
    more-likely-reached parents claim shared successors first.
    """
    G: nx.DiGraph = game["_graph"]
    root: str = game["root"]
    states: dict = game["states"]

    reachable = nx.descendants(G, root) | {root}
    sub = G.subgraph(reachable)
    topo = list(nx.topological_sort(sub))

    reach: Dict[str, float] = {s: 0.0 for s in topo}
    reach[root] = 1.0
    for s in topo:
        sd = states[s]
        if sd.get("terminal", False):
            continue
        n_prof = len(sd["transitions"])
        for t in sd["transitions"]:
            for s_next, pr in t["next"].items():
                if s_next in reach:
                    reach[s_next] += reach[s] * pr / n_prof
    return reach


def match_states(
    gameA: dict,
    keysA: dict,
    gameB: dict,
    keysB: dict,
    player_map: dict,
    c_hat: dict,
    tau: float = DEFAULT_TAU,
    weights: Optional[dict] = None,
) -> dict:
    """Match states graph-aware: only within successors of an already matched state.

    Rather than a single global assignment over all state pairs (which can pair
    states in unrelated regions of the two DAGs), matching follows the
    transition graph:

    * The two roots are matched to each other as the anchor.
    * A-states are processed in a topological order that **prefers states
      reached more likely under uniform policies** (see ``_reach_probs``).
    * When a matched state ``s → s'`` is processed, its still-unmatched
      successors in A are matched (Hungarian) against the still-unused
      successors of ``s'`` in B.

    Because states are processed parents-before-children in reach-preferred
    order and only unmatched successors are considered, a successor shared by
    several parents (DAGs need not be trees) is claimed by the first parent that
    reaches it — the more-likely-reached one.

    Per-pair similarity is unchanged:

        sim_val   = sim( [V^A_i[s]]_i ,  [V^B_{π(i)}[s']/c_hat_i]_i )
        sim_inf   = 1 − |inflow_A[s]/max_A − inflow_B[s']/max_B|
        sim_depth = 1 − |depth_A[s]/max_A − depth_B[s']/max_B|
        sim_state = w_v·sim_val + w_i·sim_inf + w_d·sim_depth

    Returns ``{stateA: stateB}`` for pairs with sim_state ≥ ``tau`` (the root
    pair is always included as the anchor).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    w_v = weights.get("v", 0.4)
    w_i = weights.get("i", 0.3)
    w_d = weights.get("d", 0.3)

    VA, VB = keysA["V"], keysB["V"]
    inA, inB = keysA["inflow"], keysB["inflow"]
    dA, dB = keysA["depth"], keysB["depth"]

    maxA_inf = max(inA.values()) + EPS
    maxB_inf = max(inB.values()) + EPS
    maxA_d = float(max(dA.values())) + 1e-6
    maxB_d = float(max(dB.values())) + 1e-6

    mp = list(player_map.items())   # [(pA, pB), …]

    def _pair_sim(sA: str, sB: str) -> float:
        vA = np.array([VA[sA][pA] for pA, _ in mp]) if mp else np.zeros(0)
        vB = (
            np.array([VB[sB][pB] / c_hat.get(pA, 1.0) for pA, pB in mp])
            if mp
            else np.zeros(0)
        )
        # Value similarity (0.5 if no players matched → neutral)
        sv = _sim(vA, vB) if mp else 0.5
        si = 1.0 - abs(inA[sA] / maxA_inf - inB[sB] / maxB_inf)
        sd_ = 1.0 - abs(dA[sA] / maxA_d - dB[sB] / maxB_d)
        return w_v * sv + w_i * si + w_d * sd_

    GA: nx.DiGraph = gameA["_graph"]
    GB: nx.DiGraph = gameB["_graph"]
    rootA, rootB = gameA["root"], gameB["root"]

    # Reach-preferred topological order over A's reachable states
    reachA = _reach_probs(gameA)
    reachableA = nx.descendants(GA, rootA) | {rootA}
    subA = GA.subgraph(reachableA)
    order = list(
        nx.lexicographical_topological_sort(
            subA, key=lambda s: (-reachA.get(s, 0.0), s)
        )
    )

    # Anchor: match the two roots to each other.
    state_map: dict = {rootA: rootB}
    used_B = {rootB}

    for sA in order:
        if sA not in state_map:
            # Unreachable through matched parents; leave for local search.
            continue
        sB = state_map[sA]
        succ_A = [c for c in GA.successors(sA) if c not in state_map]
        succ_B = [c for c in GB.successors(sB) if c not in used_B]
        if not succ_A or not succ_B:
            continue

        sim_mat = np.array([[_pair_sim(a, b) for b in succ_B] for a in succ_A])
        rows, cols = linear_sum_assignment(-sim_mat)
        for r, c in zip(rows, cols):
            if sim_mat[r, c] >= tau:
                a, b = succ_A[r], succ_B[c]
                state_map[a] = b
                used_B.add(b)

    return state_map


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Action matching
# ═════════════════════════════════════════════════════════════════════════════

def match_actions(
    gameA: dict,
    keysA: dict,
    gameB: dict,
    keysB: dict,
    state_map: dict,
    player_map: dict,
    c_hat: dict,
    tau: float = DEFAULT_TAU,
) -> dict:
    """Match actions per (matched-state, matched-player) pair via Hungarian.

    For each matched (s, s') and matched active player (i, π(i)):
        sim(Q^A_i[s][a],  Q^B_{π(i)}[s'][b] / c_hat_i)

    Returns ``action_maps[(sA, pA)] = {actionA: actionB}``
    (only pairs with sim ≥ ``tau``).
    """
    QA, QB = keysA["Q"], keysB["Q"]
    action_maps: dict = {}

    for sA, sB in state_map.items():
        sdA = gameA["states"][sA]
        sdB = gameB["states"][sB]
        # Skip terminal states (no actions)
        if sdA.get("terminal", False) or sdB.get("terminal", False):
            continue

        for pA, pB in player_map.items():
            acts_A = sdA.get("actions", {}).get(pA)
            acts_B = sdB.get("actions", {}).get(pB)
            # Both players must be active at their respective matched states
            if acts_A is None or acts_B is None:
                continue
            if pA not in QA.get(sA, {}):
                continue
            if pB not in QB.get(sB, {}):
                continue

            c = c_hat.get(pA, 1.0)
            sim_mat = np.array([
                [_sim_scalar(QA[sA][pA][a], QB[sB][pB][b] / c) for b in acts_B]
                for a in acts_A
            ])
            rows, cols = linear_sum_assignment(-sim_mat)
            mapping = {
                acts_A[r]: acts_B[c_idx]
                for r, c_idx in zip(rows, cols)
                if sim_mat[r, c_idx] >= tau
            }
            if mapping:
                action_maps[(sA, pA)] = mapping

    return action_maps


# ═════════════════════════════════════════════════════════════════════════════
# 8.  Build f0  (initial matching)
# ═════════════════════════════════════════════════════════════════════════════

def build_f0(
    gameA: dict,
    gameB: dict,
    tau: float = DEFAULT_TAU,
    weights: Optional[dict] = None,
    use_intrinsic: bool = False,
) -> dict:
    """Build the initial matching f0 through the 4-step pipeline.

    Steps
    -----
    1. Compute V, Q, inflow, depth for both games.
    2. Match players on scale-invariant structural features.
    3. Alternate (match states ↔ refine c_hat) × 3 iterations.
    4. Match actions from the stabilised state map.

    Parameters
    ----------
    use_intrinsic : if True, replace terminal payoffs with the intrinsic
        per-player power reward (see ``compute_keys`` / module docstring).

    Returns a dict with keys:
      ``player_map``, ``state_map``, ``action_maps``, ``c_hat``,
      ``c_hat_spread``, ``_keysA``, ``_keysB``.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    keysA = compute_keys(gameA, use_intrinsic=use_intrinsic)
    keysB = compute_keys(gameB, use_intrinsic=use_intrinsic)

    # Step 2 — player matching
    player_map = match_players(gameA, keysA, gameB, keysB, tau)

    # Step 3a — bootstrap scale estimate (before any state matching)
    c_hat = _bootstrap_c_hat(player_map, keysA, keysB)

    # Step 3b — alternate state matching and scale refinement (~3 rounds)
    state_map: dict = {}
    spread: dict = {}
    for _ in range(3):
        state_map = match_states(
            gameA, keysA, gameB, keysB,
            player_map, c_hat, tau, weights,
        )
        c_hat, spread = _refine_c_hat(
            player_map, state_map, keysA, keysB, c_hat
        )

    # Step 4 — action matching
    action_maps = match_actions(
        gameA, keysA, gameB, keysB,
        state_map, player_map, c_hat, tau,
    )

    return {
        "player_map": player_map,
        "state_map": state_map,
        "action_maps": action_maps,
        "c_hat": c_hat,
        "c_hat_spread": spread,
        "_keysA": keysA,
        "_keysB": keysB,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 9.  Score function
# ═════════════════════════════════════════════════════════════════════════════

def _sim_trans_profile(
    next_A: dict,
    next_B: dict,
    state_map: dict,
) -> float:
    """Compare two next-state distributions via the state map.

    Pushes A's distribution through the state map, then computes:
        1 − 0.5 × (Σ|P_A_mapped − P_B| + unmatched_A_mass + unmatched_B_mass)

    Returns a value in [0, 1].
    """
    # Accumulate A-probability mass at each mapped B-state
    matched_A: dict = {}
    unmatched_A_mass = 0.0
    for s_next, p in next_A.items():
        if s_next in state_map:
            b = state_map[s_next]
            matched_A[b] = matched_A.get(b, 0.0) + p
        else:
            unmatched_A_mass += p

    covered_B = set(matched_A)
    # B-probability mass that has no A counterpart
    unmatched_B_mass = sum(p for s_b, p in next_B.items() if s_b not in covered_B)

    # Absolute difference for every B-state that A maps to
    total_diff = sum(abs(matched_A[b] - next_B.get(b, 0.0)) for b in matched_A)

    return 1.0 - 0.5 * (total_diff + unmatched_A_mass + unmatched_B_mass)


def score(
    f: dict,
    gameA: dict,
    keysA: dict,
    gameB: dict,
    keysB: dict,
    weights: Optional[dict] = None,
) -> float:
    """Compute the overall matching score (higher is better).

    Score(f) = Σ_{(s,s')} sim_state(s,s')
             + Σ_{(s,s',i,a,b)} sim_action(s,i,a,b)
             + w_t × Σ_{(s,s')} sim_trans(s,s')

    Unmatched items contribute 0.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    w_v = weights.get("v", 0.4)
    w_i = weights.get("i", 0.3)
    w_d = weights.get("d", 0.3)
    w_t = weights.get("t", 1.5)

    player_map = f["player_map"]
    state_map = f["state_map"]
    action_maps = f["action_maps"]
    c_hat = f["c_hat"]

    VA, VB = keysA["V"], keysB["V"]
    inA, inB = keysA["inflow"], keysB["inflow"]
    dA, dB = keysA["depth"], keysB["depth"]

    maxA_inf = max(inA.values()) + EPS
    maxB_inf = max(inB.values()) + EPS
    maxA_d = float(max(dA.values())) + 1e-6
    maxB_d = float(max(dB.values())) + 1e-6
    QA, QB = keysA["Q"], keysB["Q"]

    mp = list(player_map.items())
    total = 0.0

    for sA, sB in state_map.items():
        # ── sim_state ──────────────────────────────────────────────────────────
        vA = np.array([VA[sA][pA] for pA, _ in mp]) if mp else np.zeros(0)
        vB = (
            np.array([VB[sB][pB] / c_hat.get(pA, 1.0) for pA, pB in mp])
            if mp else np.zeros(0)
        )
        sv = _sim(vA, vB) if mp else 0.5
        si = 1.0 - abs(inA[sA] / maxA_inf - inB[sB] / maxB_inf)
        sd_ = 1.0 - abs(dA[sA] / maxA_d - dB[sB] / maxB_d)
        total += w_v * sv + w_i * si + w_d * sd_

        sdA = gameA["states"][sA]
        sdB = gameB["states"][sB]
        is_term_A = sdA.get("terminal", False)
        is_term_B = sdB.get("terminal", False)

        if not is_term_A and not is_term_B:
            # ── sim_action: sum over all matched (player, action) pairs ─────────
            for pA, pB in mp:
                key = (sA, pA)
                if key in action_maps:
                    c = c_hat.get(pA, 1.0)
                    for aA, aB in action_maps[key].items():
                        q_A = QA[sA][pA][aA]
                        q_B = QB[sB][pB][aB] / c
                        total += _sim_scalar(q_A, q_B)

            # ── sim_trans: average over profiles that map through ───────────────
            active_A = sorted(sdA["actions"])
            trans_sims: list = []
            for tA in sdA["transitions"]:
                prof_A = tA["profile"]
                # Map each A-player's action to its B counterpart
                ok = True
                prof_B: dict = {}
                for pA in active_A:
                    if pA not in player_map:
                        ok = False
                        break
                    pB = player_map[pA]
                    aA = prof_A[pA]
                    amap = action_maps.get((sA, pA), {})
                    aB = amap.get(aA)
                    if aB is None:
                        ok = False
                        break
                    prof_B[pB] = aB
                if not ok:
                    continue

                # Find the corresponding B-transition
                active_B = sorted(sdB["actions"])
                # Check that prof_B covers all active B-players
                if not all(p in prof_B for p in active_B):
                    continue
                tB = next(
                    (t for t in sdB["transitions"]
                     if all(t["profile"].get(p) == prof_B[p] for p in active_B)),
                    None,
                )
                if tB is None:
                    continue

                ts = _sim_trans_profile(tA["next"], tB["next"], state_map)
                trans_sims.append(ts)

            if trans_sims:
                # Average sim_trans over mapped profiles, weight by w_t
                total += w_t * (sum(trans_sims) / len(trans_sims))

    return total


# ═════════════════════════════════════════════════════════════════════════════
# 10.  Local search  (simulated annealing over the state map)
# ═════════════════════════════════════════════════════════════════════════════

def local_search(
    f0: dict,
    gameA: dict,
    keysA: dict,
    gameB: dict,
    keysB: dict,
    weights: Optional[dict] = None,
    n_iters: int = DEFAULT_SA_ITERS,
    T0: float = DEFAULT_SA_T0,
    alpha: float = DEFAULT_SA_ALPHA,
    tau: float = DEFAULT_TAU,
    seed: Optional[int] = None,
) -> dict:
    """Improve f0 via simulated annealing over the partial injective state map.

    The player map and c_hat are kept fixed.  After each state-map move, action
    maps are re-derived via ``match_actions`` and the score is recomputed fully
    (feasible since both games have < 100 states).

    Moves
    -----
    * **reassign** — change a matched state's B-target to a different free B-state
    * **swap**     — swap the B-targets of two matched A-states
    * **add**      — match an unmatched A-state to a free B-state
    * **remove**   — drop a currently matched state pair

    SA schedule: accept improving moves always; accept worse moves with
    probability exp(ΔScore / T); T ← T × alpha each step.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    player_map = f0["player_map"]
    c_hat = f0["c_hat"]
    all_stA: List[str] = list(gameA["states"])
    all_stB: List[str] = list(gameB["states"])

    def _rebuild(sm: dict) -> dict:
        """Recompute action maps and assemble f from a state map."""
        am = match_actions(
            gameA, keysA, gameB, keysB, sm, player_map, c_hat, tau
        )
        return {
            "player_map": player_map,
            "state_map": sm,
            "action_maps": am,
            "c_hat": c_hat,
        }

    cur_f = _rebuild(dict(f0["state_map"]))
    cur_sc = score(cur_f, gameA, keysA, gameB, keysB, weights)
    best_f = cur_f
    best_sc = cur_sc
    T = T0

    for _ in range(n_iters):
        sm = dict(cur_f["state_map"])
        matched_A = list(sm)
        used_B = set(sm.values())
        free_B = [s for s in all_stB if s not in used_B]
        unmatched_A = [s for s in all_stA if s not in sm]

        # Pick a move at random (skip silently if preconditions not met)
        move = random.choice(("reassign", "swap", "add", "remove"))

        if move == "reassign" and matched_A and free_B:
            s = random.choice(matched_A)
            sm[s] = random.choice(free_B)

        elif move == "swap" and len(matched_A) >= 2:
            s1, s2 = random.sample(matched_A, 2)
            sm[s1], sm[s2] = sm[s2], sm[s1]

        elif move == "add" and unmatched_A and free_B:
            sm[random.choice(unmatched_A)] = random.choice(free_B)

        elif move == "remove" and matched_A:
            del sm[random.choice(matched_A)]

        else:
            # Preconditions not met; cool down and continue
            T *= alpha
            continue

        new_f = _rebuild(sm)
        new_sc = score(new_f, gameA, keysA, gameB, keysB, weights)
        dS = new_sc - cur_sc

        # Accept if improvement, or probabilistically if worse
        if dS > 0 or (T > 1e-12 and random.random() < math.exp(dS / T)):
            cur_f = new_f
            cur_sc = new_sc
            if cur_sc > best_sc:
                best_sc = cur_sc
                best_f = cur_f

        T *= alpha

    return best_f


# ═════════════════════════════════════════════════════════════════════════════
# 11.  End-to-end entry point
# ═════════════════════════════════════════════════════════════════════════════

def match(
    gameA: dict,
    gameB: dict,
    tau: float = DEFAULT_TAU,
    weights: Optional[dict] = None,
    sa_iters: int = DEFAULT_SA_ITERS,
    seed: Optional[int] = None,
    use_intrinsic: bool = False,
) -> dict:
    """Compute an approximate correspondence between two acyclic stochastic games.

    Parameters
    ----------
    gameA, gameB  : validated game dicts (output of ``load_game``).
    tau           : similarity threshold; pairs below this are left unmatched.
    weights       : dict with keys ``v``, ``i``, ``d``, ``t``; see module docs.
    sa_iters      : number of simulated-annealing iterations.
    seed          : optional RNG seed for reproducibility.
    use_intrinsic : if True, use the intrinsic power reward instead of real
        terminal payoffs (scale-invariant; c_hat → 1 on pure relabelling).

    Returns
    -------
    dict with keys:
      ``player_map``    {playerA: playerB}
      ``state_map``     {stateA: stateB}
      ``action_maps``   {(stateA, playerA): {actionA: actionB}}
      ``c_hat``         {playerA: estimated_scale}
      ``c_hat_spread``  {playerA: relative_std_of_ratios}  (diagnostic)
      ``score``         overall matching score (float)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    f0 = build_f0(gameA, gameB, tau, weights, use_intrinsic=use_intrinsic)
    keysA = f0["_keysA"]
    keysB = f0["_keysB"]

    f_best = local_search(
        f0, gameA, keysA, gameB, keysB,
        weights=weights, n_iters=sa_iters, tau=tau, seed=seed,
    )
    f_best["score"] = score(f_best, gameA, keysA, gameB, keysB, weights)
    f_best["c_hat_spread"] = f0.get("c_hat_spread", {})
    return f_best


# ═════════════════════════════════════════════════════════════════════════════
# 12.  __main__ demo
# ═════════════════════════════════════════════════════════════════════════════

def _make_demo_game() -> dict:
    """Build a small synthetic acyclic 2-player game for demonstration."""
    return {
        "players": ["p1", "p2"],
        "root": "s0",
        "states": {
            # s0: only p1 is active (p2 is a non-acting bystander)
            "s0": {
                "terminal": False,
                "actions": {"p1": ["L", "R"]},
                "transitions": [
                    {"profile": {"p1": "L"}, "next": {"s1": 0.7, "s2": 0.3}},
                    {"profile": {"p1": "R"}, "next": {"s2": 0.4, "s3": 0.6}},
                ],
            },
            # s1: both players active
            "s1": {
                "terminal": False,
                "actions": {"p1": ["A", "B"], "p2": ["X", "Y"]},
                "transitions": [
                    {"profile": {"p1": "A", "p2": "X"}, "next": {"s4": 1.0}},
                    {"profile": {"p1": "A", "p2": "Y"}, "next": {"s5": 1.0}},
                    {"profile": {"p1": "B", "p2": "X"}, "next": {"s4": 0.5, "s5": 0.5}},
                    {"profile": {"p1": "B", "p2": "Y"}, "next": {"s6": 1.0}},
                ],
            },
            # s2: only p2 is active
            "s2": {
                "terminal": False,
                "actions": {"p2": ["U", "V", "W"]},
                "transitions": [
                    {"profile": {"p2": "U"}, "next": {"s4": 0.8, "s5": 0.2}},
                    {"profile": {"p2": "V"}, "next": {"s5": 0.5, "s6": 0.5}},
                    {"profile": {"p2": "W"}, "next": {"s6": 1.0}},
                ],
            },
            # s3: both players active (one action each)
            "s3": {
                "terminal": False,
                "actions": {"p1": ["C"], "p2": ["Z"]},
                "transitions": [
                    {"profile": {"p1": "C", "p2": "Z"}, "next": {"s6": 1.0}},
                ],
            },
            # Terminals
            "s4": {"terminal": True, "payoffs": {"p1":  3.0, "p2":  1.0}},
            "s5": {"terminal": True, "payoffs": {"p1":  1.0, "p2":  2.0}},
            "s6": {"terminal": True, "payoffs": {"p1": -1.0, "p2":  0.5}},
        },
    }


def _relabel_and_rescale(
    game: dict,
    state_rename: dict,
    player_rename: dict,
    action_rename: dict,
    payoff_scales: dict,
) -> dict:
    """Return a copy of *game* with renamed states/players/actions and
    rescaled payoffs.  Used to build game B for the demo.

    Parameters
    ----------
    state_rename  : {old_state: new_state}
    player_rename : {old_player: new_player}
    action_rename : {old_action: new_action}  (flat; actions must be globally unique)
    payoff_scales : {old_player: scale_factor}
    """
    p_map = player_rename
    s_map = state_rename
    a_map = action_rename

    new_states: dict = {}
    for s, sd in game["states"].items():
        ns = s_map[s]
        if sd.get("terminal", False):
            new_states[ns] = {
                "terminal": True,
                "payoffs": {
                    p_map[p]: v * payoff_scales.get(p, 1.0)
                    for p, v in sd["payoffs"].items()
                },
            }
        else:
            new_actions = {
                p_map[p]: [a_map[a] for a in acts]
                for p, acts in sd["actions"].items()
            }
            new_trans = []
            for t in sd["transitions"]:
                new_prof = {p_map[p]: a_map[a] for p, a in t["profile"].items()}
                new_next = {s_map[sn]: pr for sn, pr in t["next"].items()}
                new_trans.append({"profile": new_prof, "next": new_next})
            new_states[ns] = {
                "terminal": False,
                "actions": new_actions,
                "transitions": new_trans,
            }

    return {
        "players": [p_map[p] for p in game["players"]],
        "root": s_map[game["root"]],
        "states": new_states,
    }


if __name__ == "__main__":
    import pprint

    random.seed(42)
    np.random.seed(42)

    # ── Build game A ─────────────────────────────────────────────────────────
    raw_A = _make_demo_game()
    gameA = load_game(raw_A)
    print("=== Game A ===")
    print("Players:", gameA["players"])
    print("States :", list(gameA["states"]))

    # ── Build game B by relabelling + rescaling ───────────────────────────────
    true_scales = {"p1": 2.5, "p2": 0.4}   # B multiplies p1 by 2.5, p2 by 0.4

    state_rename  = {"s0": "t0", "s1": "t1", "s2": "t2",
                     "s3": "t3", "s4": "t4", "s5": "t5", "s6": "t6"}
    player_rename = {"p1": "q1", "p2": "q2"}
    action_rename = {
        "L": "LL", "R": "RR",
        "A": "AA", "B": "BB", "X": "XX", "Y": "YY",
        "U": "UU", "V": "VV", "W": "WW",
        "C": "CC", "Z": "ZZ",
    }

    raw_B = _relabel_and_rescale(
        raw_A, state_rename, player_rename, action_rename, true_scales
    )
    gameB = load_game(raw_B)
    print("\n=== Game B  (relabelled + rescaled) ===")
    print("Players:", gameB["players"])
    print("States :", list(gameB["states"]))

    # ── Run the pipeline ─────────────────────────────────────────────────────
    print("\nRunning match pipeline …")
    result = match(gameA, gameB, tau=0.4, sa_iters=20_000, seed=42)

    print("\n=== Results ===")
    print("Player map   :", result["player_map"])
    print("State map    :", result["state_map"])
    print("c_hat        :", {k: f"{v:.4f}" for k, v in result["c_hat"].items()})
    print("c_hat_spread :", {k: f"{v:.4f}" for k, v in result["c_hat_spread"].items()})
    print("Score        :", f"{result['score']:.4f}")

    print("\nAction maps:")
    for (s, p), amap in sorted(result["action_maps"].items()):
        print(f"  ({s}, {p}): {amap}")

    print("\nTrue scales  :", {f"p{i+1}": v for i, v in enumerate(true_scales.values())})
    print("True state map:", {v: state_rename[v] for v in state_rename})
    print("True player map:", player_rename)

    # ── Run again with intrinsic power rewards ────────────────────────────────
    print("\n" + "=" * 60)
    print("Re-running with use_intrinsic=True (power metric, scale-free)")
    print("=" * 60)
    result_intr = match(gameA, gameB, tau=0.4, sa_iters=20_000,
                        seed=42, use_intrinsic=True)
    print("Player map   :", result_intr["player_map"])
    print("State map    :", result_intr["state_map"])
    print("c_hat        :", {k: f"{v:.4f}" for k, v in result_intr["c_hat"].items()})
    print("c_hat_spread :", {k: f"{v:.4f}" for k, v in result_intr["c_hat_spread"].items()})
    print("Score        :", f"{result_intr['score']:.4f}")
    print("(c_hat ≈ 1.0 expected since intrinsic rewards are payoff-scale-free)")
