"""
╔══════════════════════════════════════════════════════════════╗
║   BAYESIAN STATE-SPACE CRICKET PREDICTION ENGINE v1.0        ║
║   Architecture: Thread-Isolated Producer-Consumer            ║
║   Math Stack: Eigenvectors · HMM · Markov Chains · Softmax  ║
╚══════════════════════════════════════════════════════════════╝

Install dependencies:
    pip install numpy scipy

Run:
    python cricket_engine.py
"""

import sys
import time
import math
import random
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import linalg


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class LivePayload:
    """Raw state vector arriving from the API producer thread."""
    runs: int
    wickets: int
    balls_remaining: int
    striker_id: int
    bowler_id: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class MathResult:
    """Processed output pushed from the math consumer thread to the UI thread."""
    win_probability: float          # 0.0 – 100.0
    delta: float                    # momentum trend vs. previous tick
    outcome_probs: np.ndarray       # [P(0), P(1), P(2), P(3), P(4), P(6), P(W)]
    expected_score: float
    runs: int
    wickets: int
    balls_remaining: int
    active_ops: list[str]           # labels shown in the "Math Running" panel


# ─────────────────────────────────────────────────────────────
# PHASE 1 — HISTORICAL PRIOR ENGINE
# ─────────────────────────────────────────────────────────────

class HistoricalPriorEngine:
    """
    Calculates the pre-match mathematical baseline.

    Steps
    -----
    1. Eigenvector Centrality  → λ (team dominance)
    2. Glicko-2 style σ        → team volatility
    3. HMM Viterbi             → form multiplier
    4. Bradley-Terry           → P(A wins) with venue/toss bias
    """

    TEAM_NAMES: list[str] = [
        "Mumbai Indians", "Chennai Super Kings", "Royal Challengers",
        "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
        "Punjab Kings", "Sunrisers Hyderabad", "Gujarat Titans", "LSG"
    ]

    def __init__(self, team_a_idx: int = 0, team_b_idx: int = 1,
                 toss_winner: int = 0, venue_bias: float = 0.04) -> None:
        self.team_a_idx = team_a_idx
        self.team_b_idx = team_b_idx
        self.toss_winner = toss_winner
        self.venue_bias = venue_bias

        self._adjacency: np.ndarray = self._build_adjacency()
        self._sigma: np.ndarray = self._build_volatility()
        self._lambda: np.ndarray = self._compute_eigenvector_centrality()
        self._form_multipliers: np.ndarray = self._run_hmm_viterbi()
        self.prior_p_a: float = self._bradley_terry()

    # ── 1. Adjacency Matrix ───────────────────────────────────

    def _build_adjacency(self) -> np.ndarray:
        """
        Mock head-to-head win-rate adjacency matrix (10×10).
        A[i][j] = fraction of times team i beat team j historically.
        """
        rng = np.random.default_rng(seed=42)
        raw = rng.uniform(0.3, 0.7, size=(10, 10))
        np.fill_diagonal(raw, 0.0)
        # Normalize rows so they are proper distributions
        row_sums = raw.sum(axis=1, keepdims=True)
        return raw / row_sums

    # ── 2. Glicko-2 style volatility ─────────────────────────

    def _build_volatility(self) -> np.ndarray:
        """
        σ[i] ∈ [0.5, 1.5] represents consistency; lower σ = more predictable.
        """
        rng = np.random.default_rng(seed=7)
        return rng.uniform(0.5, 1.5, size=10)

    # ── 3. Eigenvector Centrality via scipy.linalg.eig ───────

    def _compute_eigenvector_centrality(self) -> np.ndarray:
        """
        Solve A·v = λ·v.  The principal eigenvector gives each team's
        true dominance score regardless of schedule strength.
        """
        eigenvalues, eigenvectors = linalg.eig(self._adjacency)
        # Principal eigenvector = column corresponding to largest real eigenvalue
        principal_idx = np.argmax(eigenvalues.real)
        v = eigenvectors[:, principal_idx].real
        # Normalise to positive unit vector
        v = np.abs(v)
        v /= v.sum()
        return v

    # ── 4. Hidden Markov Model — pure-NumPy Baum-Welch + Viterbi ──

    @staticmethod
    def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Univariate Gaussian probability density (vectorised)."""
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    def _baum_welch(
        self,
        obs: np.ndarray,          # shape (T,)
        n_states: int = 2,
        n_iter: int = 40,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Baum-Welch EM for a 2-state Gaussian HMM — pure NumPy, no C compiler.

        Returns
        -------
        pi  : initial state distribution  (n_states,)
        A   : transition matrix            (n_states, n_states)
        mu  : Gaussian means               (n_states,)
        sig : Gaussian std-devs            (n_states,)
        """
        T = len(obs)
        rng = np.random.default_rng(seed)

        # ── Initialise parameters ─────────────────────────────
        pi  = np.ones(n_states) / n_states
        A   = rng.dirichlet(np.ones(n_states), size=n_states)   # row-stochastic
        # Initialise means at percentile splits of the observations
        mu  = np.percentile(obs, np.linspace(20, 80, n_states))
        sig = np.full(n_states, obs.std() + 1e-6)

        for _ in range(n_iter):
            # ── E-step: Forward-Backward ──────────────────────
            B = np.column_stack([
                self._gaussian_pdf(obs, mu[s], sig[s]) for s in range(n_states)
            ])  # (T, n_states)

            # Forward pass
            alpha = np.zeros((T, n_states))
            alpha[0] = pi * B[0]
            alpha[0] /= alpha[0].sum() + 1e-300
            scales = np.zeros(T)
            scales[0] = alpha[0].sum()

            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[t]
                scales[t] = alpha[t].sum() + 1e-300
                alpha[t] /= scales[t]

            # Backward pass
            beta = np.zeros((T, n_states))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (A * B[t + 1] * beta[t + 1]).sum(axis=1)
                beta[t] /= beta[t].sum() + 1e-300

            # Posterior state probabilities  γ[t, s]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # Two-slice marginals  ξ[t, i, j]
            xi = np.zeros((T - 1, n_states, n_states))
            for t in range(T - 1):
                xi[t] = (
                    alpha[t, :, None]
                    * A
                    * B[t + 1, None, :]
                    * beta[t + 1, None, :]
                )
                xi[t] /= xi[t].sum() + 1e-300

            # ── M-step: update parameters ─────────────────────
            pi  = gamma[0] / (gamma[0].sum() + 1e-300)
            A   = xi.sum(axis=0) / (xi.sum(axis=0).sum(axis=1, keepdims=True) + 1e-300)
            mu  = (gamma * obs[:, None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)
            var = (gamma * (obs[:, None] - mu) ** 2).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)
            sig = np.sqrt(var) + 1e-6

        return pi, A, mu, sig

    def _viterbi(
        self,
        obs: np.ndarray,
        pi: np.ndarray,
        A: np.ndarray,
        mu: np.ndarray,
        sig: np.ndarray,
    ) -> np.ndarray:
        """
        Viterbi algorithm — returns the most-likely hidden state sequence.
        All arithmetic in log-space for numerical stability.
        """
        T = len(obs)
        n_states = len(pi)

        log_A  = np.log(A  + 1e-300)
        log_pi = np.log(pi + 1e-300)

        # Log-emission matrix
        log_B = np.column_stack([
            np.log(self._gaussian_pdf(obs, mu[s], sig[s]) + 1e-300)
            for s in range(n_states)
        ])  # (T, n_states)

        delta   = np.full((T, n_states), -np.inf)
        psi     = np.zeros((T, n_states), dtype=int)

        delta[0] = log_pi + log_B[0]

        for t in range(1, T):
            for s in range(n_states):
                scores      = delta[t - 1] + log_A[:, s]
                psi[t, s]   = int(np.argmax(scores))
                delta[t, s] = scores[psi[t, s]] + log_B[t, s]

        # Backtrack
        states      = np.zeros(T, dtype=int)
        states[-1]  = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def _run_hmm_viterbi(self) -> np.ndarray:
        """
        2-state Gaussian HMM per team — pure NumPy (no hmmlearn).
        State 0 = Out-of-Form,  State 1 = In-Form.
        Returns a form multiplier ∈ [0.85, 1.15] for each team.
        """
        multipliers = np.ones(10)
        rng = np.random.default_rng(seed=13)

        for i in range(10):
            # Mock recent match scores for team i (last 8 matches)
            base_score = 160 + rng.integers(-20, 20)
            scores = rng.normal(loc=base_score, scale=15, size=8)

            # Fit HMM with Baum-Welch
            pi, A, mu, sig = self._baum_welch(scores, n_states=2, n_iter=40, seed=42 + i)

            # Decode with Viterbi
            state_seq = self._viterbi(scores, pi, A, mu, sig)

            # In-Form state = whichever state has the higher mean
            in_form_state = int(np.argmax(mu))

            # Fraction of recent matches spent In-Form → [0.85, 1.15]
            form_fraction = float(np.mean(state_seq == in_form_state))
            multipliers[i] = 0.85 + form_fraction * 0.30

        return multipliers

    # ── 5. Bradley-Terry pre-match probability ───────────────

    def _bradley_terry(self) -> float:
        """
        P(A wins) = exp(λ_A/σ_A) / (exp(λ_A/σ_A) + exp(λ_B/σ_B))
        Adjusted by HMM form multiplier and venue/toss bias.
        """
        a, b = self.team_a_idx, self.team_b_idx

        score_a = math.exp(self._lambda[a] / self._sigma[a]) * self._form_multipliers[a]
        score_b = math.exp(self._lambda[b] / self._sigma[b]) * self._form_multipliers[b]

        p_a = score_a / (score_a + score_b)

        # Toss winner batting first gets a small edge
        if self.toss_winner == a:
            p_a = min(p_a + self.venue_bias, 0.99)
        else:
            p_a = max(p_a - self.venue_bias, 0.01)

        return p_a

    # ── Public accessors ──────────────────────────────────────

    @property
    def team_a_name(self) -> str:
        return self.TEAM_NAMES[self.team_a_idx]

    @property
    def team_b_name(self) -> str:
        return self.TEAM_NAMES[self.team_b_idx]

    @property
    def lambda_scores(self) -> np.ndarray:
        return self._lambda

    @property
    def sigma_scores(self) -> np.ndarray:
        return self._sigma


# ─────────────────────────────────────────────────────────────
# PHASE 2 — LIVE DATA PRODUCER (API intake thread)
# ─────────────────────────────────────────────────────────────

def api_intake_worker(
    data_queue: queue.Queue,
    stop_event: threading.Event,
    total_balls: int = 120
) -> None:
    """
    Daemon thread: simulates polling a live JSON API every 3-5 seconds.
    Pushes sanitised LivePayload objects onto data_queue.
    Drops corrupted frames silently and holds state.
    """
    runs = 0
    wickets = 0
    balls_bowled = 0
    BATSMEN_POOL = list(range(1, 12))
    BOWLERS_POOL = list(range(101, 112))

    striker_id = random.choice(BATSMEN_POOL)
    bowler_id = random.choice(BOWLERS_POOL)

    while not stop_event.is_set():
        time.sleep(random.uniform(3.0, 5.0))

        balls_bowled += 1
        balls_remaining = total_balls - balls_bowled

        # ── Simulate a delivery outcome ───────────────────────
        outcome_weights = [30, 20, 12, 5, 10, 8, 15]  # 0,1,2,3,4,6,W
        outcome = random.choices([0, 1, 2, 3, 4, 6, -1], weights=outcome_weights)[0]

        if outcome == -1:
            wickets += 1
            striker_id = random.choice([b for b in BATSMEN_POOL if b != striker_id])
        else:
            runs += outcome

        # Occasionally rotate bowler every ~6 balls
        if balls_bowled % 6 == 0:
            bowler_id = random.choice(BOWLERS_POOL)

        # ── Inject a corrupt frame ~8% of the time ────────────
        if random.random() < 0.08:
            # Drop; hold state — do not push to queue
            continue

        # ── Sanitisation gate ─────────────────────────────────
        if balls_remaining < 0 or runs < 0 or wickets < 0 or wickets > 10:
            continue   # malformed — discard

        payload = LivePayload(
            runs=runs,
            wickets=wickets,
            balls_remaining=balls_remaining,
            striker_id=striker_id,
            bowler_id=bowler_id,
        )
        data_queue.put(payload)

        if balls_remaining <= 0 or wickets >= 10:
            stop_event.set()
            break


# ─────────────────────────────────────────────────────────────
# PHASE 3 — LIVE MATH CHURN (consumer thread)
# ─────────────────────────────────────────────────────────────

def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(z - z.max())
    return e / e.sum()


def quantitative_worker(
    data_queue: queue.Queue,
    display_queue: queue.Queue,
    stop_event: threading.Event,
    prior: HistoricalPriorEngine,
    target: int,
) -> None:
    """
    Daemon thread: consumes LivePayload, runs the full math pipeline,
    and pushes MathResult onto display_queue.
    """
    # Per-player mock parameters (μ, θ_batsman, δ_bowler)
    rng = np.random.default_rng(seed=99)
    batsman_theta: dict[int, float] = {pid: rng.uniform(-0.5, 1.2) for pid in range(1, 12)}
    bowler_delta:  dict[int, float] = {pid: rng.uniform(-0.8, 0.8) for pid in range(101, 112)}

    mu_base = np.array([0.50, 0.85, 0.40, 0.15, 0.55, 0.40, 0.90])  # baseline μ per outcome
    prev_win_prob = prior.prior_p_a * 100.0

    while not stop_event.is_set() or not data_queue.empty():
        try:
            payload: LivePayload = data_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        ops: list[str] = []

        # ── A. Micro-matchup Z-scores (Log-Odds additive model) ───
        ops.append("Log-Odds Z-Score [striker vs bowler]")
        theta = batsman_theta.get(payload.striker_id, 0.0)
        delta = bowler_delta.get(payload.bowler_id, 0.0)
        z_scores: np.ndarray = mu_base + theta - delta   # shape (7,)

        # ── B. Softmax → probability distribution ────────────────
        ops.append("Softmax normalisation → P[0,1,2,3,4,6,W]")
        outcome_probs: np.ndarray = _softmax(z_scores)

        # ── C. Absorbing Markov Chain: T^B ──────────────────────
        ops.append(f"Absorbing Markov T^{payload.balls_remaining} (matrix power)")
        # Build a compact 2-state transition matrix:
        # State 0 = batting continues, State 1 = wicket (absorbing)
        p_wicket = outcome_probs[6]
        p_survive = 1.0 - p_wicket

        T = np.array([
            [p_survive, p_wicket],
            [0.0,       1.0     ]
        ])

        B = max(payload.balls_remaining, 1)
        T_pow = np.linalg.matrix_power(
            T, min(B, 60)   # cap to avoid numerical explosion
        )

        # Expected fraction of remaining balls survived
        survival_prob = float(T_pow[0, 0])

        # Expected runs from here: weight outcomes (0,1,2,3,4,6) by their probs
        run_weights = np.array([0, 1, 2, 3, 4, 6])
        expected_run_rate = float(np.dot(outcome_probs[:6], run_weights)) * survival_prob
        expected_score = payload.runs + expected_run_rate * payload.balls_remaining

        # ── D. Win Probability ────────────────────────────────────
        ops.append("Bradley-Terry prior × live score projection")
        runs_needed = target - payload.runs
        balls_left = max(payload.balls_remaining, 1)
        wkts_left = 10 - payload.wickets

        # Logistic adjustment on top of prior
        required_rr = runs_needed / (balls_left / 6.0)
        current_rr  = expected_run_rate * 6.0

        logistic_delta = 1.0 / (1.0 + math.exp(-(current_rr - required_rr) * 0.4))
        # Blend prior with live logistic signal (prior decays with balls bowled)
        balls_bowled = 120 - payload.balls_remaining
        prior_weight = max(0.05, 1.0 - balls_bowled / 120.0)
        live_weight  = 1.0 - prior_weight

        win_prob = (prior.prior_p_a * prior_weight + logistic_delta * live_weight)
        # Wickets remaining penalty
        wkt_penalty = math.exp(-0.15 * (10 - wkts_left))
        win_prob = min(max(win_prob * wkt_penalty, 0.01), 0.99) * 100.0

        momentum = win_prob - prev_win_prob
        prev_win_prob = win_prob

        result = MathResult(
            win_probability=win_prob,
            delta=momentum,
            outcome_probs=outcome_probs,
            expected_score=expected_score,
            runs=payload.runs,
            wickets=payload.wickets,
            balls_remaining=payload.balls_remaining,
            active_ops=ops,
        )
        display_queue.put(result)
        data_queue.task_done()


# ─────────────────────────────────────────────────────────────
# PHASE 4 — ZERO-LAG CLI DASHBOARD (main thread)
# ─────────────────────────────────────────────────────────────

ANSI = {
    "clear":   "\033[H\033[J",
    "bold":    "\033[1m",
    "reset":   "\033[0m",
    "cyan":    "\033[96m",
    "green":   "\033[92m",
    "red":     "\033[91m",
    "yellow":  "\033[93m",
    "magenta": "\033[95m",
    "white":   "\033[97m",
    "dim":     "\033[2m",
    "blue":    "\033[94m",
}

def _bar(prob: float, width: int = 28) -> str:
    """Return a coloured ASCII progress bar for a probability 0–1."""
    filled = int(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    colour = ANSI["green"] if prob > 0.55 else ANSI["yellow"] if prob > 0.40 else ANSI["red"]
    return f"{colour}{bar}{ANSI['reset']}"


def terminal_renderer(
    display_queue: queue.Queue,
    stop_event: threading.Event,
    prior: HistoricalPriorEngine,
    team_a: str,
    team_b: str,
    target: int,
) -> None:
    """
    Main thread: reads MathResult from display_queue and redraws the
    terminal dashboard in-place using ANSI escape sequences.
    """
    OUTCOME_LABELS = ["  0 ", "  1 ", "  2 ", "  3 ", "  4 ", "  6 ", "  W "]
    OUTCOME_COLS   = ["dim","green","green","yellow","cyan","magenta","red"]
    last_result: Optional[MathResult] = None
    tick = 0

    # ── Loading animation while math threads spin up ──────────
    spinners = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    for s in spinners * 3:
        sys.stdout.write(f"\r  {ANSI['cyan']}{s}{ANSI['reset']}  Initialising HMM priors & eigenvector engine…")
        sys.stdout.flush()
        time.sleep(0.10)
    sys.stdout.write(ANSI["clear"])

    while not stop_event.is_set() or not display_queue.empty():
        try:
            result: MathResult = display_queue.get(timeout=0.5)
            last_result = result
            tick += 1
        except queue.Empty:
            if last_result is None:
                continue
            result = last_result

        wp = result.win_probability
        delta = result.delta
        balls_bowled = 120 - result.balls_remaining
        overs_done = f"{balls_bowled // 6}.{balls_bowled % 6}"
        total_overs = "20.0"
        runs_needed = max(target - result.runs, 0)
        rr_req = runs_needed / max(result.balls_remaining / 6.0, 0.1)

        delta_str = (
            f"{ANSI['green']}▲ +{delta:+.2f}%{ANSI['reset']}"
            if delta >= 0
            else f"{ANSI['red']}▼ {delta:+.2f}%{ANSI['reset']}"
        )

        # ── Build dashboard ────────────────────────────────────
        W = 68
        border_top    = f"{'═' * W}"
        border_mid    = f"{'─' * W}"
        border_bottom = f"{'═' * W}"

        lines: list[str] = []
        a = lambda *parts: lines.append("".join(parts))

        a(ANSI["cyan"], ANSI["bold"], border_top, ANSI["reset"])
        a(ANSI["cyan"], ANSI["bold"],
          f"  BAYESIAN STATE-SPACE CRICKET ENGINE  ",
          ANSI["dim"], f"  tick #{tick:04d}   {time.strftime('%H:%M:%S')}",
          ANSI["reset"])
        a(ANSI["cyan"], border_mid, ANSI["reset"])

        # ── Match header ──────────────────────────────────────
        a(ANSI["bold"], ANSI["white"],
          f"  {team_a:<28}  vs  {team_b:>26}", ANSI["reset"])
        a(ANSI["dim"],
          f"  Prior P(A): {prior.prior_p_a*100:.1f}%   Target: {target}   "
          f"Overs: {overs_done}/{total_overs}", ANSI["reset"])
        a(ANSI["cyan"], border_mid, ANSI["reset"])

        # ── Live scoreboard ───────────────────────────────────
        a(ANSI["bold"],
          f"  SCORE  {ANSI['yellow']}{result.runs}/{result.wickets}{ANSI['reset']}{ANSI['bold']}"
          f"   Need {ANSI['cyan']}{runs_needed}{ANSI['reset']}{ANSI['bold']}"
          f" off {result.balls_remaining} balls"
          f"   RR reqd {ANSI['magenta']}{rr_req:.2f}{ANSI['reset']}")
        a("")

        # ── Win probability bar ───────────────────────────────
        p = wp / 100.0
        a(ANSI["bold"], f"  {team_a:<22}", ANSI["reset"],
          f"  {_bar(p)}  ",
          ANSI["bold"], f"{wp:5.1f}%", ANSI["reset"],
          f"  {delta_str}")
        a(ANSI["bold"], f"  {team_b:<22}", ANSI["reset"],
          f"  {_bar(1-p)}  ",
          ANSI["bold"], f"{(100-wp):5.1f}%", ANSI["reset"])
        a("")

        # ── Per-outcome probability panel ─────────────────────
        a(ANSI["cyan"], border_mid, ANSI["reset"])
        a(ANSI["dim"], "  BALL OUTCOME DISTRIBUTION (this delivery)", ANSI["reset"])
        a("")
        row_label = "  "
        row_bar   = "  "
        for idx, (label, col_key, prob) in enumerate(
                zip(OUTCOME_LABELS, OUTCOME_COLS, result.outcome_probs)):
            w = 6
            filled = int(prob * w * 3)
            bar_str = "▓" * min(filled, w) + "░" * (w - min(filled, w))
            row_label += f"{ANSI[col_key]}{label}{ANSI['reset']} "
            row_bar   += f"{ANSI[col_key]}{bar_str}{ANSI['reset']} "
        a(row_label)
        a(row_bar)
        probs_str = "  " + " ".join(
            f"{ANSI[c]}{p*100:4.1f}%{ANSI['reset']}"
            for c, p in zip(OUTCOME_COLS, result.outcome_probs)
        )
        a(probs_str)
        a("")

        # ── Expected score ────────────────────────────────────
        a(ANSI["dim"], f"  Expected final score (Markov projection): ",
          ANSI["yellow"], ANSI["bold"], f"{result.expected_score:.1f}", ANSI["reset"])
        a("")

        # ── Active math operations ────────────────────────────
        a(ANSI["cyan"], border_mid, ANSI["reset"])
        a(ANSI["dim"], "  MATHEMATICAL OPERATIONS (background threads)", ANSI["reset"])
        for op in result.active_ops:
            a(f"   {ANSI['green']}✔{ANSI['reset']}  {ANSI['dim']}{op}{ANSI['reset']}")
        a("")

        # ── Eigenvector snapshot ──────────────────────────────
        la = prior.lambda_scores[prior.team_a_idx]
        lb = prior.lambda_scores[prior.team_b_idx]
        sa = prior.sigma_scores[prior.team_a_idx]
        sb = prior.sigma_scores[prior.team_b_idx]
        a(ANSI["dim"],
          f"  λ({team_a[:12]}): {la:.4f}  σ: {sa:.3f}   "
          f"λ({team_b[:12]}): {lb:.4f}  σ: {sb:.3f}",
          ANSI["reset"])
        a(ANSI["cyan"], border_bottom, ANSI["reset"])
        a(ANSI["dim"],
          "  [Ctrl+C to quit]   Threads: API-Producer · Math-Consumer · UI-Main",
          ANSI["reset"])

        # ── Render ────────────────────────────────────────────
        sys.stdout.write(ANSI["clear"])
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        display_queue.task_done()

    # ── Match over ────────────────────────────────────────────
    sys.stdout.write(ANSI["clear"])
    final_wp = last_result.win_probability if last_result else 50.0
    winner = team_a if final_wp > 50 else team_b
    sys.stdout.write(
        f"\n  {ANSI['bold']}{ANSI['green']}MATCH COMPLETE  — Predicted winner: {winner} "
        f"({final_wp:.1f}%){ANSI['reset']}\n\n"
    )
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{ANSI['cyan']}{ANSI['bold']}  Bootstrapping Bayesian priors (HMM + Eigenvector)…{ANSI['reset']}")
    print(f"  {ANSI['dim']}This may take a few seconds.{ANSI['reset']}\n")

    # ── Match configuration ───────────────────────────────────
    TEAM_A_IDX  = 0   # Mumbai Indians
    TEAM_B_IDX  = 1   # Chennai Super Kings
    TOSS_WINNER = 0
    TARGET      = 178  # Team B chasing

    prior = HistoricalPriorEngine(
        team_a_idx=TEAM_A_IDX,
        team_b_idx=TEAM_B_IDX,
        toss_winner=TOSS_WINNER,
        venue_bias=0.04
    )

    # ── Shared queues ─────────────────────────────────────────
    data_queue:    queue.Queue = queue.Queue(maxsize=50)
    display_queue: queue.Queue = queue.Queue(maxsize=50)
    stop_event = threading.Event()

    # ── Thread 1: API Producer (daemon) ───────────────────────
    producer = threading.Thread(
        target=api_intake_worker,
        args=(data_queue, stop_event, 120),
        daemon=True,
        name="Thread-API-Producer"
    )

    # ── Thread 2: Math Consumer (daemon) ──────────────────────
    consumer = threading.Thread(
        target=quantitative_worker,
        args=(data_queue, display_queue, stop_event, prior, TARGET),
        daemon=True,
        name="Thread-Math-Consumer"
    )

    producer.start()
    consumer.start()

    # ── Main thread: UI renderer ──────────────────────────────
    try:
        terminal_renderer(
            display_queue=display_queue,
            stop_event=stop_event,
            prior=prior,
            team_a=prior.team_a_name,
            team_b=prior.team_b_name,
            target=TARGET,
        )
    except KeyboardInterrupt:
        stop_event.set()
        print(f"\n{ANSI['yellow']}  Engine stopped by user.{ANSI['reset']}\n")

    producer.join(timeout=3)
    consumer.join(timeout=3)


if __name__ == "__main__":
    main()
