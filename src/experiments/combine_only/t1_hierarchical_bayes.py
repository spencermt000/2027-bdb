"""T1: hierarchical Bayesian model on attempt features (spec 1.1).

Question: given a player's body and forty profile, what shuttle performance is expected, and how far does each attempt
deviate? The model splits variation into between-player differences, systematic attempt-order change, and attempt noise.

Targets (standardized per fit):  cod_cost (attempt level); brake_dist, dwell, reaccel_ratio, entry_ratio (plant level,
two rows per attempt with plant_idx and, except for entry_ratio, entry_speed as covariates).
Likelihood: Student-t. mu = alpha + c'beta + u_pos[i] + u_i + gamma * (attempt_order - 1) [+ delta * plant + eta * entry].
Variants: V1 linear; V2 correlated random intercept + random entry_speed slope per player (plant targets);
V3 all targets jointly (LKJ player-effect correlations, multivariate-t correlated residuals).

Scores: out-of-fold. A held-out player's score is relative to the expectation given c alone (u_i drawn from its prior).
In-sample fits supply player effects, coefficients, gamma, variance decomposition and PSIS-LOO.
Column names follow the shared schema, so extras start with `score_` (e.g. `score_player_effect_<target>`).

Usage: python src/experiments/combine_only/t1_hierarchical_bayes.py [--fast] [--no-cv] [--targets cod_cost ...]
       [--variants V1 V2 V3]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "process"))
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402
from scipy.stats import kstest  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

numpyro.set_platform("cpu")

from bio import Conditioner, feature_columns  # noqa: E402
from config import load_yaml, paths, pipeline_cfg, seed_everything  # noqa: E402
from runlog import Timer, write_report  # noqa: E402
from schema import write_scores  # noqa: E402

KEYS = ["nfl_id", "event_id", "attempt"]
TARGETS = {
    "cod_cost": dict(level="attempt", cols=["cod_cost"], entry=False),
    "brake_dist": dict(level="plant", cols=["brake_dist_1", "brake_dist_2"], entry=True),
    "dwell": dict(level="plant", cols=["dwell_1", "dwell_2"], entry=True),
    "reaccel_ratio": dict(level="plant", cols=["reaccel_ratio_1", "reaccel_ratio_2"], entry=True),
    "entry_ratio": dict(level="plant", cols=["entry_ratio_1", "entry_ratio_2"], entry=False),
}
JOINT_DIMS = ["cod_cost"] + [f"{t}_{k}" for t in ("brake_dist", "dwell", "reaccel_ratio", "entry_ratio") for k in (1, 2)]
JOINT_ENTRY = {d: (d.split("_")[-1] if d.rsplit("_", 1)[0] in ("brake_dist", "dwell", "reaccel_ratio") else None) for d in JOINT_DIMS}


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------

def load_data(population: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Attempts (conditioning + targets + fold) for the chosen population, folds table, position labels."""
    p = paths()
    cond = pd.read_parquet(p["derived"] / "conditioning.parquet")
    att = pd.read_parquet(p["derived"] / "shuttle_attempt_features.parquet")
    folds = pd.read_parquet(p["derived"] / "folds.parquet")
    tcols = sorted({c for t in TARGETS.values() for c in t["cols"]} | {"entry_speed_1", "entry_speed_2", "t_ref_total", "qc_plants_ok"})
    df = cond.merge(att[KEYS + tcols], on=KEYS, how="left")
    keep = df["eval_ok"] if population == "eval_ok" else df["qc_plants_ok"].fillna(False).astype(bool)
    df = df[keep].merge(folds[["nfl_id", "fold"]], on="nfl_id").reset_index(drop=True)
    pos_cols = [c for c in df.columns if c.startswith("pos_c_")]
    labels = [c[len("pos_c_"):] for c in pos_cols]
    df["pos"] = df[pos_cols].to_numpy().argmax(axis=1)
    return df, folds, labels


def model_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Conditioning columns for T1: position enters as a random effect, so the position one-hots are dropped."""
    numeric, binary = feature_columns(df)
    return numeric, [c for c in binary if not c.startswith(("pos_c_", "pos_n_"))]


def keep_columns(Xa: pd.DataFrame, tol: float = 0.9999) -> np.ndarray:
    """Mask of columns to model: drop constants and exact duplicates (|corr| > tol with an earlier kept column).

    f40_t90 is exactly tau * ln(10), for example. Decided on the training rows only and reused for held-out rows.
    """
    keep = (Xa.std(axis=0) > 0).to_numpy().copy()
    X = Xa.to_numpy(float)
    for j in range(X.shape[1]):
        if not keep[j]:
            continue
        for i in range(j):
            if keep[i] and abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) > tol:
                keep[j] = False
                break
    return keep


class Design:
    """Row-level arrays for one fit or one scoring set. Rows are attempts (attempt-level target) or plants (stacked)."""

    def __init__(self, df: pd.DataFrame, target: str, cond: Conditioner, stats: dict | None = None, players: np.ndarray | None = None,
                 keep_const: np.ndarray | None = None):
        spec = TARGETS[target]
        self.target, self.spec = target, spec
        self.df = df.reset_index(drop=True)
        self.players = np.unique(self.df["nfl_id"]) if players is None else players
        pmap = {p: i for i, p in enumerate(self.players)}
        Xa = cond.transform(self.df)
        self.keep = keep_columns(Xa) if keep_const is None else keep_const
        Xa = Xa.loc[:, self.keep]
        self.columns = list(Xa.columns)
        if spec["level"] == "attempt":
            reps = [(0, spec["cols"][0], None)]
        else:
            reps = [(0, spec["cols"][0], "entry_speed_1"), (1, spec["cols"][1], "entry_speed_2")]
        parts = []
        for plant, ycol, ecol in reps:
            part = pd.DataFrame({"attempt_row": np.arange(len(self.df)), "plant": float(plant), "y_raw": self.df[ycol].to_numpy(float),
                                 "entry_raw": self.df[ecol].to_numpy(float) if (ecol and spec["entry"]) else np.nan})
            parts.append(part)
        rows = pd.concat(parts, ignore_index=True)
        ar = rows["attempt_row"].to_numpy()
        self.attempt_row = ar
        self.X = Xa.to_numpy(float)[ar]
        self.player = self.df["nfl_id"].map(pmap).to_numpy()[ar]
        self.pos = self.df["pos"].to_numpy()[ar]
        self.order = (self.df["attempt_order"].to_numpy(float) - 1.0)[ar]
        self.plant = rows["plant"].to_numpy() if spec["level"] == "plant" else None
        if stats is None:
            stats = {"y_mean": np.nanmean(rows["y_raw"]), "y_sd": np.nanstd(rows["y_raw"])}
            if spec["entry"]:
                stats.update(e_mean=np.nanmean(rows["entry_raw"]), e_sd=np.nanstd(rows["entry_raw"]))
        self.stats = stats
        self.y_raw = rows["y_raw"].to_numpy()
        self.y = (self.y_raw - stats["y_mean"]) / stats["y_sd"]
        self.entry = (rows["entry_raw"].to_numpy() - stats["e_mean"]) / stats["e_sd"] if spec["entry"] else None
        self.n_players, self.n = len(self.players), len(self.y)


# ---------------------------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------------------------

def t1_model(X, player, pos, order, n_players, n_pos, pri, plant=None, entry=None, slope=False, y=None):
    """V1 (slope=False) and V2 (slope=True) for attempt- or plant-level targets."""
    p = X.shape[1]
    cs, ss = pri["coef_sd"], pri["scale_sd"]
    alpha = numpyro.sample("alpha", dist.Normal(0.0, cs))
    beta = numpyro.sample("beta", dist.Normal(0.0, cs).expand([p]))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, cs))
    sigma_player = numpyro.sample("sigma_player", dist.HalfNormal(ss))
    sigma_pos = numpyro.sample("sigma_pos", dist.HalfNormal(ss))
    sigma_attempt = numpyro.sample("sigma_attempt", dist.HalfNormal(ss))
    nu = numpyro.sample("nu", dist.Gamma(pri["nu_concentration"], pri["nu_rate"]))
    z_pos = numpyro.sample("z_pos", dist.Normal(0.0, 1.0).expand([n_pos]))
    mu = alpha + X @ beta + sigma_pos * z_pos[pos] + gamma * order
    if plant is not None:
        mu = mu + numpyro.sample("delta", dist.Normal(0.0, cs)) * plant
    if entry is not None:
        mu = mu + numpyro.sample("eta", dist.Normal(0.0, cs)) * entry
    if slope:
        sigma_slope = numpyro.sample("sigma_slope", dist.HalfNormal(ss))
        L = numpyro.sample("L_player", dist.LKJCholesky(2, pri["lkj_eta"]))
        z = numpyro.sample("z_player", dist.Normal(0.0, 1.0).expand([n_players, 2]))
        eff = (z @ L.T) * jnp.stack([sigma_player, sigma_slope])
        mu = mu + eff[player, 0] + eff[player, 1] * entry
    else:
        z = numpyro.sample("z_player", dist.Normal(0.0, 1.0).expand([n_players]))
        mu = mu + sigma_player * z[player]
    numpyro.sample("y", dist.StudentT(nu, mu, sigma_attempt), obs=y)


def joint_model(X, player, pos, order, E, n_players, n_pos, pri, y=None):
    """V3: D-dimensional outcome per attempt, LKJ-correlated player effects, correlated multivariate-t residuals."""
    n, p = X.shape
    D = len(JOINT_DIMS)
    cs, ss = pri["coef_sd"], pri["scale_sd"]
    alpha = numpyro.sample("alpha", dist.Normal(0.0, cs).expand([D]))
    beta = numpyro.sample("beta", dist.Normal(0.0, cs).expand([p, D]))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, cs).expand([D]))
    eta = numpyro.sample("eta", dist.Normal(0.0, cs).expand([D]))
    sigma_pos = numpyro.sample("sigma_pos", dist.HalfNormal(ss).expand([D]))
    z_pos = numpyro.sample("z_pos", dist.Normal(0.0, 1.0).expand([n_pos, D]))
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(ss).expand([D]))
    L_u = numpyro.sample("L_u", dist.LKJCholesky(D, pri["lkj_eta"]))
    z = numpyro.sample("z_player", dist.Normal(0.0, 1.0).expand([n_players, D]))
    u = (z @ L_u.T) * sigma_u
    sigma_e = numpyro.sample("sigma_e", dist.HalfNormal(ss).expand([D]))
    L_e = numpyro.sample("L_e", dist.LKJCholesky(D, pri["lkj_eta"]))
    nu = numpyro.sample("nu", dist.Gamma(pri["nu_concentration"], pri["nu_rate"]))
    mu = alpha + X @ beta + sigma_pos * z_pos[pos] + gamma * order[:, None] + eta * E + u[player]
    numpyro.sample("y", dist.MultivariateStudentT(nu, mu, scale_tril=sigma_e[:, None] * L_e), obs=y)


def run_mcmc(model, kwargs: dict, mcfg: dict, seed: int):
    """NUTS with the configured chains; returns (samples grouped by chain as numpy, diagnostics dict)."""
    chains = mcfg["chains"]
    method = "parallel" if jax.local_device_count() >= chains else "sequential"
    mc = MCMC(NUTS(model, target_accept_prob=mcfg["target_accept"]), num_warmup=mcfg["warmup"], num_samples=mcfg["draws"],
              num_chains=chains, chain_method=method, progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), **kwargs)
    grouped = {k: np.asarray(v) for k, v in mc.get_samples(group_by_chain=True).items()}
    div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    return grouped, {"divergences": div, "chains": chains, "draws": mcfg["draws"]}


def diagnostics(grouped: dict, keys: list[str]) -> dict:
    """Max R-hat and min bulk ESS over the scalar/low-dimensional parameters in `keys` (ArviZ)."""
    import arviz as az
    sub = {k: grouped[k] for k in keys if k in grouped and grouped[k].ndim <= 3}
    summ = az.summary(az.from_dict(posterior=sub), round_to=4)
    return {"rhat_max": float(summ["r_hat"].max()), "ess_bulk_min": float(summ["ess_bulk"].min()),
            "ess_tail_min": float(summ["ess_tail"].min())}


def flat(grouped: dict) -> dict:
    """(chains, draws, ...) -> (chains*draws, ...)."""
    return {k: v.reshape((-1,) + v.shape[2:]) for k, v in grouped.items()}


# ---------------------------------------------------------------------------------------------------------------
# posterior-predictive machinery (numpy; V1 / V2)
# ---------------------------------------------------------------------------------------------------------------

def mu_common(S: dict, d: Design) -> np.ndarray:
    """(draws, rows): expectation given c alone (population part; no player effect, no random slope)."""
    mu = S["alpha"][:, None] + S["beta"] @ d.X.T + S["sigma_pos"][:, None] * S["z_pos"][:, d.pos] + S["gamma"][:, None] * d.order[None]
    if d.plant is not None:
        mu = mu + S["delta"][:, None] * d.plant[None]
    if d.entry is not None:
        mu = mu + S["eta"][:, None] * d.entry[None]
    return mu


def has_slope(S: dict) -> bool:
    return "sigma_slope" in S


def player_effects(S: dict, d: Design) -> tuple[np.ndarray, np.ndarray | None]:
    """In-sample (draws, players) intercept effects and slopes (None for V1)."""
    if has_slope(S):
        sig = np.stack([S["sigma_player"], S["sigma_slope"]], axis=1)                       # (S, 2)
        eff = np.einsum("snj,skj->snk", S["z_player"], S["L_player"]) * sig[:, None, :]    # z @ L.T
        return eff[..., 0], eff[..., 1]
    return S["sigma_player"][:, None] * S["z_player"], None


def new_player_effect(S: dict, d: Design, rng: np.random.Generator) -> np.ndarray:
    """(draws, rows) effect of an unseen player drawn from the prior: intercept (+ slope * entry for V2). One draw per row."""
    n_s, n = len(S["alpha"]), d.n
    if has_slope(S):
        z = rng.standard_normal((n_s, n, 2))
        sig = np.stack([S["sigma_player"], S["sigma_slope"]], axis=1)
        eff = np.einsum("snj,skj->snk", z, S["L_player"]) * sig[:, None, :]
        return eff[..., 0] + eff[..., 1] * d.entry[None]
    return S["sigma_player"][:, None] * rng.standard_normal((n_s, n))


def marginal_scale(S: dict, d: Design) -> np.ndarray:
    """(draws, rows) new-player predictive scale: sqrt(var of player effect + sigma_attempt^2)."""
    var = S["sigma_player"][:, None] ** 2 + S["sigma_attempt"][:, None] ** 2
    if has_slope(S):
        rho = S["L_player"][:, 1, 0]
        var = (S["sigma_player"][:, None] ** 2 + (S["sigma_slope"][:, None] * d.entry[None]) ** 2
               + 2 * rho[:, None] * S["sigma_player"][:, None] * S["sigma_slope"][:, None] * d.entry[None] + S["sigma_attempt"][:, None] ** 2)
    return np.sqrt(var)


def y_rep_new(S: dict, d: Design, rng: np.random.Generator, mu_c: np.ndarray | None = None) -> np.ndarray:
    mu_c = mu_common(S, d) if mu_c is None else mu_c
    eps = rng.standard_t(S["nu"][:, None], size=mu_c.shape) * S["sigma_attempt"][:, None]
    return mu_c + new_player_effect(S, d, rng) + eps


def pit_values(y_rep: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Posterior-predictive PIT of each observed y: share of draws below it (ties count half)."""
    return (y_rep < y[None]).mean(axis=0) + 0.5 * (y_rep == y[None]).mean(axis=0)


def heldout_player_ll(S: dict, d: Design, n_draws: int, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """(players,) log marginal predictive density of each held-out player's rows, integrating the player effect over its prior.

    Gauss-Hermite quadrature over the random effect(s) per posterior draw, then log-mean-exp over draws. Standardized units.
    """
    idx = rng.choice(len(S["alpha"]), size=min(n_draws, len(S["alpha"])), replace=False)
    Sx = {k: v[idx] for k, v in S.items()}
    mu_c = mu_common(Sx, d)
    nodes, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    logw = np.log(w / w.sum())
    out = np.zeros(d.n_players)
    slope = has_slope(Sx)
    if slope:
        nodes2 = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2)
        logw2 = (logw[:, None] + logw[None, :]).reshape(-1)
    for j in range(d.n_players):
        rows = np.flatnonzero(d.player == j)
        if slope:
            sig = np.stack([Sx["sigma_player"], Sx["sigma_slope"]], axis=1)                         # (S, 2)
            eff = np.einsum("kj,sij->ski", nodes2, Sx["L_player"]) * sig[:, None, :]                # (S, K, 2)
            m = mu_c[:, None, rows] + eff[:, :, 0:1] + eff[:, :, 1:2] * d.entry[None, None, rows]    # (S, K, r)
            lw = logw2
        else:
            eff = Sx["sigma_player"][:, None] * nodes[None, :]                                      # (S, K)
            m = mu_c[:, None, rows] + eff[:, :, None]
            lw = logw
        sc = Sx["sigma_attempt"][:, None, None]
        ll = student_t.logpdf((d.y[rows][None, None, :] - m) / sc, Sx["nu"][:, None, None]) - np.log(sc)
        per_draw = logsumexp(ll.sum(axis=2) + lw[None, :], axis=1)
        out[j] = logsumexp(per_draw) - np.log(len(idx))
    return out


def pointwise_loglik(S: dict, d: Design) -> np.ndarray:
    """(draws, rows) log-likelihood of each observed row given its own player's effect (for PSIS-LOO)."""
    u, s = player_effects(S, d)
    mu = mu_common(S, d) + u[:, d.player]
    if s is not None:
        mu = mu + s[:, d.player] * d.entry[None]
    sc = S["sigma_attempt"][:, None]
    return student_t.logpdf((d.y[None] - mu) / sc, S["nu"][:, None]) - np.log(sc)


def psis_loo(grouped: dict, d: Design, S: dict) -> dict:
    import arviz as az
    ll = pointwise_loglik(S, d)
    n_chain, n_draw = grouped["alpha"].shape[:2]
    idata = az.from_dict(posterior={"alpha": grouped["alpha"]}, log_likelihood={"y": ll.reshape(n_chain, n_draw, -1)})
    r = az.loo(idata, pointwise=True)
    return {"elpd_loo": float(r.elpd_loo), "se": float(r.se), "p_loo": float(r.p_loo), "n_bad_k": int((np.asarray(r.pareto_k) > 0.7).sum()),
            "n_obs": int(d.n)}


# ---------------------------------------------------------------------------------------------------------------
# fitting + scoring one target (V1 / V2)
# ---------------------------------------------------------------------------------------------------------------

def fit_target(df: pd.DataFrame, target: str, cond_cols: tuple, n_pos: int, variant: str, cfg: dict, mcfg: dict, seed: int):
    """Fit on all rows of `df`. Returns (design, grouped samples, flat samples, diagnostics)."""
    cond = Conditioner(*cond_cols).fit(df)
    d = Design(df, target, cond)
    kw = dict(X=d.X, player=d.player, pos=d.pos, order=d.order, n_players=d.n_players, n_pos=n_pos, pri=cfg["priors"], plant=d.plant,
              entry=d.entry, slope=(variant == "V2"), y=d.y)
    grouped, diag = run_mcmc(t1_model, kw, mcfg, seed)
    keys = ["alpha", "beta", "gamma", "sigma_player", "sigma_pos", "sigma_attempt", "nu", "delta", "eta", "sigma_slope"]
    diag.update(diagnostics(grouped, keys))
    return d, cond, grouped, flat(grouped), diag


def score_design(S: dict, d: Design, rng: np.random.Generator, scale_to_raw: bool = True) -> dict:
    """Scores for the rows of `d` under posterior samples S (new-player expectation given c)."""
    mu_c = mu_common(S, d)
    y_rep = y_rep_new(S, d, rng, mu_c)
    z = (d.y[None] - mu_c) / marginal_scale(S, d)
    return {"mu_c": mu_c, "z_mean": z.mean(axis=0), "z_sd": z.std(axis=0), "pit": pit_values(y_rep, d.y),
            "pred_median_raw": np.median(mu_c, axis=0) * d.stats["y_sd"] + d.stats["y_mean"]}


def fold_scores(df: pd.DataFrame, target: str, cond_cols, n_pos: int, variant: str, cfg: dict, mcfg: dict, seed: int, rng) -> pd.DataFrame:
    """Out-of-fold scores for every row of the target; one grouped-CV pass over the shared folds."""
    pieces = []
    for k in sorted(df["fold"].unique()):
        tr, te = df[df["fold"] != k], df[df["fold"] == k]
        d_tr, cond, grouped, S, _ = fit_target(tr, target, cond_cols, n_pos, variant, cfg, mcfg, seed + 100 + int(k))
        d_te = Design(te, target, cond, stats=d_tr.stats, keep_const=d_tr.keep)
        sc = score_design(S, d_te, rng)
        ll = heldout_player_ll(S, d_te, cfg["heldout"]["n_draws"], cfg["heldout"]["n_nodes"], rng)
        ll_raw = ll - np.bincount(d_te.player, minlength=d_te.n_players) * np.log(d_tr.stats["y_sd"])
        n_att = te.groupby("nfl_id").size().reindex(d_te.players).to_numpy()
        pieces.append(pd.DataFrame({"attempt_row": te.index[d_te.attempt_row], "plant": d_te.plant if d_te.plant is not None else 0.0,
                                    "z": sc["z_mean"], "z_sd": sc["z_sd"], "pit": sc["pit"], "pred_raw": sc["pred_median_raw"],
                                    "ll_player_per_attempt": (ll_raw / n_att)[d_te.player]}))
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------
# V3 joint model
# ---------------------------------------------------------------------------------------------------------------

class JointDesign:
    def __init__(self, df: pd.DataFrame, cond: Conditioner, stats: dict | None = None, players=None, keep_const=None):
        self.df = df.reset_index(drop=True)
        self.players = np.unique(self.df["nfl_id"]) if players is None else players
        pmap = {p: i for i, p in enumerate(self.players)}
        Xa = cond.transform(self.df)
        self.keep = keep_columns(Xa) if keep_const is None else keep_const
        self.columns = list(Xa.columns[self.keep])
        self.X = Xa.to_numpy(float)[:, self.keep]
        Y = np.column_stack([self.df[d].to_numpy(float) for d in JOINT_DIMS])
        E = np.column_stack([self.df[f"entry_speed_{JOINT_ENTRY[d]}"].to_numpy(float) if JOINT_ENTRY[d] else np.zeros(len(self.df)) for d in JOINT_DIMS])
        if stats is None:
            stats = {"y_mean": Y.mean(axis=0), "y_sd": Y.std(axis=0)}
            stats["e_mean"] = np.array([E[:, j].mean() if JOINT_ENTRY[d] else 0.0 for j, d in enumerate(JOINT_DIMS)])
            stats["e_sd"] = np.array([E[:, j].std() if JOINT_ENTRY[d] else 1.0 for j, d in enumerate(JOINT_DIMS)])
        self.stats = stats
        self.y_raw = Y
        self.y = (Y - stats["y_mean"]) / stats["y_sd"]
        self.E = np.where(np.array([bool(JOINT_ENTRY[d]) for d in JOINT_DIMS])[None], (E - stats["e_mean"]) / stats["e_sd"], 0.0)
        self.player = self.df["nfl_id"].map(pmap).to_numpy()
        self.pos = self.df["pos"].to_numpy()
        self.order = self.df["attempt_order"].to_numpy(float) - 1.0
        self.n, self.n_players = len(self.df), len(self.players)


def joint_mu_common(S: dict, d: JointDesign) -> np.ndarray:
    """(draws, n, D) expectation given c alone."""
    return (S["alpha"][:, None, :] + np.einsum("np,spd->snd", d.X, S["beta"]) + S["sigma_pos"][:, None, :] * S["z_pos"][:, d.pos, :]
            + S["gamma"][:, None, :] * d.order[None, :, None] + S["eta"][:, None, :] * d.E[None])


def joint_player_u(S: dict) -> np.ndarray:
    """In-sample (draws, players, D) player effects."""
    return np.einsum("snd,skd->snk", S["z_player"], S["L_u"]) * S["sigma_u"][:, None, :]


def mvt_logpdf(r: np.ndarray, L_inv: np.ndarray, logdet: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Multivariate-t log density. r (..., D) residuals; L_inv (D, D) inverse Cholesky of the scale; logdet = log|Sigma|."""
    D = r.shape[-1]
    m = np.sum(np.einsum("ij,...j->...i", L_inv, r) ** 2, axis=-1)
    return (gammaln((nu + D) / 2) - gammaln(nu / 2) - D / 2 * np.log(nu * np.pi) - 0.5 * logdet - (nu + D) / 2 * np.log1p(m / nu))


def fit_joint(df: pd.DataFrame, cond_cols: tuple, n_pos: int, cfg: dict, mcfg: dict, seed: int):
    cond = Conditioner(*cond_cols).fit(df)
    d = JointDesign(df, cond)
    kw = dict(X=d.X, player=d.player, pos=d.pos, order=d.order, E=d.E, n_players=d.n_players, n_pos=n_pos, pri=cfg["priors"], y=d.y)
    grouped, diag = run_mcmc(joint_model, kw, mcfg, seed)
    diag.update(diagnostics(grouped, ["gamma", "sigma_u", "sigma_e", "nu", "sigma_pos"]))
    return d, cond, grouped, flat(grouped), diag


def joint_scale_chol(S: dict) -> np.ndarray:
    return S["sigma_e"][:, :, None] * S["L_e"]


def joint_fold_scores(df: pd.DataFrame, cond_cols, n_pos: int, cfg: dict, mcfg: dict, seed: int, rng) -> pd.DataFrame:
    """Out-of-fold V3: z / PIT per dimension (cod_cost) and joint held-out log-likelihood per player."""
    pieces = []
    D = len(JOINT_DIMS)
    for k in sorted(df["fold"].unique()):
        tr, te = df[df["fold"] != k], df[df["fold"] == k]
        d_tr, cond, _, S, _ = fit_joint(tr, cond_cols, n_pos, cfg, mcfg, seed + 100 + int(k))
        d = JointDesign(te, cond, stats=d_tr.stats, keep_const=d_tr.keep)
        n_s = len(S["alpha"])
        mu_c = joint_mu_common(S, d)
        scale_u = np.einsum("sij,sj->sij", S["L_u"], S["sigma_u"])
        cov_u = np.einsum("sij,skj->sik", scale_u, scale_u)
        Lc = joint_scale_chol(S)
        cov_e = np.einsum("sij,skj->sik", Lc, Lc)
        # cod_cost (dim 0) signed standardized residual and PIT, new-player predictive
        u_new = np.einsum("sij,snj->sni", scale_u, rng.standard_normal((n_s, d.n, D)))
        g = rng.chisquare(S["nu"][:, None], size=(n_s, d.n)) / S["nu"][:, None]
        eps = np.einsum("sij,snj->sni", Lc, rng.standard_normal((n_s, d.n, D))) / np.sqrt(g)[..., None]
        y_rep = mu_c + u_new + eps
        scale0 = np.sqrt(cov_u[:, 0, 0] + cov_e[:, 0, 0])
        z = (d.y[None, :, 0] - mu_c[:, :, 0]) / scale0[:, None]
        pit = np.stack([pit_values(y_rep[:, :, j], d.y[:, j]) for j in range(D)], axis=1)
        # joint held-out ll by Monte Carlo over the player effect
        idx = rng.choice(n_s, size=min(cfg["heldout"]["n_draws"], n_s), replace=False)
        K = cfg["heldout"]["n_mc"]
        ll_player = np.zeros(d.n_players)
        zk = rng.standard_normal((K, D))
        for j in range(d.n_players):
            rows = np.flatnonzero(d.player == j)
            per_draw = []
            for s in idx:
                u = zk @ scale_u[s].T                                          # (K, D)
                L_inv = np.linalg.inv(Lc[s])
                logdet = 2 * np.sum(np.log(np.diag(Lc[s])))
                r = d.y[rows][None] - (mu_c[s, rows][None] + u[:, None, :])    # (K, r, D)
                lp = mvt_logpdf(r, L_inv, logdet, S["nu"][s]).sum(axis=1)
                per_draw.append(logsumexp(lp) - np.log(K))
            ll_player[j] = logsumexp(per_draw) - np.log(len(idx))
        ll_raw = ll_player - np.bincount(d.player, minlength=d.n_players) * np.sum(np.log(d_tr.stats["y_sd"]))
        n_att = te.groupby("nfl_id").size().reindex(d.players).to_numpy()
        piece = pd.DataFrame({"attempt_row": te.index, "z": z.mean(axis=0), "z_sd": z.std(axis=0), "ll_player_per_attempt": (ll_raw / n_att)[d.player],
                              "pred_raw": np.median(mu_c[:, :, 0], axis=0) * d_tr.stats["y_sd"][0] + d_tr.stats["y_mean"][0]})
        for j, name in enumerate(JOINT_DIMS):
            piece[f"pit_{name}"] = pit[:, j]
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------------------------------------------

def summarize(x: np.ndarray, q=(5, 95)) -> dict:
    return {"mean": float(np.mean(x)), "sd": float(np.std(x)), f"q{q[0]}": float(np.percentile(x, q[0])), f"q{q[1]}": float(np.percentile(x, q[1]))}


def attempts_per_player(d: Design) -> np.ndarray:
    """Attempts (not rows) per player; plant-level designs have two rows per attempt."""
    return np.bincount(d.player[d.plant == 0] if d.plant is not None else d.player, minlength=d.n_players)


def cohort_summary(S: dict, d: Design, columns: list[str]) -> dict:
    """Variance decomposition, gamma, coefficients, shrinkage for one V1/V2 fit (original target units for scales)."""
    sd_y = d.stats["y_sd"]
    icc = S["sigma_player"] ** 2 / (S["sigma_player"] ** 2 + S["sigma_attempt"] ** 2)
    out = {"icc": summarize(icc), "gamma": summarize(S["gamma"] * sd_y), "sigma_player": summarize(S["sigma_player"] * sd_y),
           "sigma_attempt": summarize(S["sigma_attempt"] * sd_y), "sigma_pos": summarize(S["sigma_pos"] * sd_y), "nu": summarize(S["nu"]),
           "coef": {c: summarize(S["beta"][:, i] * sd_y) for i, c in enumerate(columns)}}
    for k in ("delta", "eta"):
        if k in S:
            out[k] = summarize(S[k] * sd_y)
    u, slope = player_effects(S, d)
    n_per = attempts_per_player(d)
    shrink = 1.0 - u.var(axis=0) / np.mean(S["sigma_player"] ** 2)       # share of prior variance resolved by the data
    out["shrinkage"] = {int(n): float(np.mean(shrink[n_per == n])) for n in np.unique(n_per)}
    if slope is not None:
        out["slope"] = {"sigma_slope": summarize(S["sigma_slope"] * sd_y), "eta_pop": summarize(S["eta"] * sd_y),
                        "corr_intercept_slope": summarize(S["L_player"][:, 1, 0]),
                        "player_slope_sd_of_means": float(np.std(slope.mean(axis=0)) * sd_y)}
    return out


def calibration(pit: np.ndarray) -> dict:
    h, _ = np.histogram(pit, bins=10, range=(0, 1))
    return {"ks_p": float(kstest(pit, "uniform").pvalue), "hist": h.tolist(), "n": int(len(pit)),
            "cover50": float(np.mean((pit > 0.25) & (pit < 0.75))), "cover90": float(np.mean((pit > 0.05) & (pit < 0.95)))}


# ---------------------------------------------------------------------------------------------------------------
# outputs: attempt_scores.parquet, report.md, figures
# ---------------------------------------------------------------------------------------------------------------

def insample_scores(d: Design, S: dict, rng: np.random.Generator) -> dict:
    """Full-fit scores plus the within-player attempt deviation (y - expectation - own player effect) / sigma_attempt."""
    sc = score_design(S, d, rng)
    u, slope = player_effects(S, d)
    own = u[:, d.player] + (slope[:, d.player] * d.entry[None] if slope is not None else 0.0)
    sc["dev"] = ((d.y[None] - sc["mu_c"] - own) / S["sigma_attempt"][:, None]).mean(axis=0)
    sc["u_mean"] = u.mean(axis=0) * d.stats["y_sd"]
    sc["u_sd"] = u.std(axis=0) * d.stats["y_sd"]
    return sc


def _attempt_col(values: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    out[rows] = values
    return out


def build_scores(df: pd.DataFrame, results: dict, full_fits: dict, cfg: dict, rng) -> pd.DataFrame | None:
    n = len(df)
    out = df[KEYS].copy()
    out["exp_id"] = "T1"
    primary = cfg["primary_variant"]
    pk = f"{primary}/cod_cost" if primary != "V3" else "V3/joint"
    have_cv = pk in results["cv"]
    if pk not in full_fits and not have_cv:
        print("cod_cost was not fitted; no attempt_scores.parquet written")
        return None
    if have_cv:
        cv = results["cv"][pk].sort_values("attempt_row")
        z, zsd = cv["z"].to_numpy(), cv["z_sd"].to_numpy()
        rows = cv["attempt_row"].to_numpy()
    else:
        d, S = full_fits[pk]
        sc = insample_scores(d, S, rng)
        z, zsd, rows = sc["z_mean"], sc["z_sd"], np.arange(n)
    out["score_primary"] = _attempt_col(-z, rows, n)                       # cost: lower is better, so the sign is flipped
    out["score_uncertainty"] = _attempt_col(zsd, rows, n)
    for key, (d, S) in full_fits.items():
        variant, target = key.split("/")
        if variant == "V3":
            continue
        sc = insample_scores(d, S, rng)
        ar = d.attempt_row
        if d.plant is None:
            out[f"score_player_effect_{target}"] = _attempt_col(sc["u_mean"][d.player], ar, n)
            out[f"score_player_effect_sd_{target}"] = _attempt_col(sc["u_sd"][d.player], ar, n)
            if variant == primary:
                out["score_primary_insample"] = _attempt_col(-sc["z_mean"], ar, n)
                out["score_attempt_dev_cod_cost"] = _attempt_col(-sc["dev"], ar, n)
        elif variant == primary or f"{primary}/{target}" not in full_fits:
            out[f"score_player_effect_{target}"] = _attempt_col(sc["u_mean"][d.player][d.plant == 0], ar[d.plant == 0], n)
            out[f"score_player_effect_sd_{target}"] = _attempt_col(sc["u_sd"][d.player][d.plant == 0], ar[d.plant == 0], n)
        if variant == primary:
            src = results["cv"].get(key)
            if src is not None:
                if d.plant is None:
                    out[f"score_pit_{target}"] = _attempt_col(src.sort_values("attempt_row")["pit"].to_numpy(), np.sort(src["attempt_row"].to_numpy()), n)
                    out["score_ll_heldout" if target == "cod_cost" else f"score_ll_heldout_{target}"] = _attempt_col(
                        src.sort_values("attempt_row")["ll_player_per_attempt"].to_numpy(), np.sort(src["attempt_row"].to_numpy()), n)
                else:
                    for k in (0, 1):
                        sub = src[src["plant"] == k].sort_values("attempt_row")
                        out[f"score_pit_{target}_p{k + 1}"] = _attempt_col(sub["pit"].to_numpy(), sub["attempt_row"].to_numpy(), n)
                    sub = src[src["plant"] == 0].sort_values("attempt_row")
                    out[f"score_ll_heldout_{target}"] = _attempt_col(sub["ll_player_per_attempt"].to_numpy(), sub["attempt_row"].to_numpy(), n)
    if "V3/joint" in results["cv"] and primary != "V3":
        j = results["cv"]["V3/joint"].sort_values("attempt_row")
        out["score_primary_v3"] = _attempt_col(-j["z"].to_numpy(), j["attempt_row"].to_numpy(), n)
        out["score_ll_heldout_v3"] = _attempt_col(j["ll_player_per_attempt"].to_numpy(), j["attempt_row"].to_numpy(), n)
    for c in out.columns:
        if c.startswith("score_"):
            out[c] = out[c].astype(float)
    return out


def _pct(x) -> str:
    return f"{x:.3f}"


def _ci(s: dict, nd=3) -> str:
    return f"{s['mean']:.{nd}f} [{s['q5']:.{nd}f}, {s['q95']:.{nd}f}]"


def build_outputs(df, results, full_fits, cfg, seed, runtime, pos_labels, do_cv, fast) -> None:
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = paths()
    out_dir = p["outputs"] / "T1"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 7)
    scores = build_scores(df, results, full_fits, cfg, rng)
    if scores is not None:
        write_scores(scores, "T1")

    B = []
    B += ["## Run", "", f"- Population: `{cfg['population']}`; attempts {len(df)}, players {df['nfl_id'].nunique()}; positions: {pos_labels}",
          f"- MCMC: {cfg['fast_mcmc'] if fast else cfg['mcmc']}" + (" (**fast smoke-test settings**)" if fast else ""),
          f"- Scores are {'out-of-fold on the shared grouped folds (held-out `u_i` drawn from its prior)' if do_cv else '**in-sample** (cv disabled)'}.",
          "- Extra columns carry the `score_` prefix required by the shared schema: `score_player_effect_<target>`, `score_pit_<target>[_p1/_p2]`,"
          " `score_ll_heldout[_<target>]`, `score_primary_insample`, `score_attempt_dev_cod_cost` (attempt vs the player's own level), `score_primary_v3`.", ""]

    B += ["## Convergence (full-data fits)", "", "| fit | R-hat max | ESS bulk min | ESS tail min | divergences |", "|---|---|---|---|---|"]
    for k, dg in results["diag"].items():
        flag = "" if dg["rhat_max"] < cfg["rhat_max"] else " **>1.01**"
        B.append(f"| {k} | {dg['rhat_max']:.4f}{flag} | {dg['ess_bulk_min']:.0f} | {dg['ess_tail_min']:.0f} | {dg['divergences']} |")
    B.append("")

    coh = results["cohort"]
    if coh:
        B += ["## Variance decomposition and attempt-order effect", "",
              "Model ICC = sigma_player^2 / (sigma_player^2 + sigma_attempt^2). Scales in the target's own units. gamma = change per attempt number.", "",
              "| fit | model ICC [90%] | sigma_player | sigma_attempt | sigma_pos | gamma [90%] | nu |", "|---|---|---|---|---|---|---|"]
        for k, c in coh.items():
            B.append(f"| {k} | {_ci(c['icc'])} | {c['sigma_player']['mean']:.3f} | {c['sigma_attempt']['mean']:.3f} | {c['sigma_pos']['mean']:.3f} | "
                     f"{_ci(c['gamma'])} | {c['nu']['mean']:.1f} |")
        B += ["", "## Shrinkage of player effects", "",
              "Share of the prior variance of `u_i` resolved by the data, by the player's number of attempts (0 = fully shrunk to the mean).", "",
              "| fit | " + " | ".join(f"{n} attempt(s)" for n in (1, 2, 3)) + " |", "|---|---|---|---|"]
        for k, c in coh.items():
            sh = c["shrinkage"]
            B.append(f"| {k} | " + " | ".join(f"{sh.get(n, float('nan')):.2f}" for n in (1, 2, 3)) + " |")
        B += ["", "## Coefficients (largest posterior mean / SD per fit, original target units, 90% interval)", ""]
        for k, c in coh.items():
            if not k.startswith("V1/"):
                continue
            top = sorted(c["coef"].items(), key=lambda kv: -abs(kv[1]["mean"]) / (kv[1]["sd"] + 1e-12))[:6]
            B.append(f"**{k}** " + "; ".join(f"`{n}` {_ci(s, 3)}" for n, s in top))
            B.append("")
            extra = [f"{nm} {_ci(c[nm])}" for nm in ("delta", "eta") if nm in c]
            if extra:
                B += ["Plant/entry terms: " + "; ".join(extra), ""]
        v2 = {k: c for k, c in coh.items() if "slope" in c}
        if v2:
            B += ["## V2: player-specific speed-vs-braking slopes", "",
                  "Entry speed is chosen by the athlete, so the slope mixes ability with strategy: read it as a trade-off, not a causal effect.", "",
                  "| target | population slope eta [90%] | sigma_slope | corr(intercept, slope) | SD of player slope means |", "|---|---|---|---|---|"]
            for k, c in v2.items():
                sl = c["slope"]
                B.append(f"| {k} | {_ci(sl['eta_pop'])} | {sl['sigma_slope']['mean']:.3f} | {_ci(sl['corr_intercept_slope'], 2)} | {sl['player_slope_sd_of_means']:.3f} |")
            B.append("")

    if results["loo"]:
        B += ["## PSIS-LOO (full data, observation level)", "",
              "Leave-one-observation-out, so rows of a player inform each other: optimistic compared with the grouped held-out likelihood below.", "",
              "| fit | elpd_loo | SE | p_loo | Pareto k > 0.7 | rows |", "|---|---|---|---|---|---|"]
        for k, r in results["loo"].items():
            B.append(f"| {k} | {r['elpd_loo']:.1f} | {r['se']:.1f} | {r['p_loo']:.1f} | {r['n_bad_k']} | {r['n_obs']} |")
        B.append("")

    cv = results["cv"]
    if cv:
        B += ["## Held-out predictive log-likelihood (grouped folds, original target units)", "",
              "Per attempt: the held-out player's joint log predictive density (player effect integrated over its prior) divided by their attempt count.", "",
              "| fit | mean ll per attempt |", "|---|---|"]
        for k, v in cv.items():
            B.append(f"| {k} | {v.groupby('attempt_row')['ll_player_per_attempt'].first().mean():.3f} |")
        plant_targets = [t for t in TARGETS if TARGETS[t]["level"] == "plant"]
        parts = {v: [results["cv"].get(f"{v}/{t}") for t in ["cod_cost"] + plant_targets] for v in ("V1", "V2")}
        if "V3/joint" in cv and all(x is not None for x in parts["V1"]):
            ind = sum(x.groupby("attempt_row")["ll_player_per_attempt"].first().mean() for x in parts["V1"])
            B += ["", f"Joint V3 vs the sum of independent V1 fits (same 9 outcomes): **V3 {cv['V3/joint'].groupby('attempt_row')['ll_player_per_attempt'].first().mean():.3f}** "
                  f"vs V1 sum {ind:.3f}. (V3 uses a Monte Carlo integral over the 9-d player effect, so it is noisier.)"]
        B.append("")

        B += ["## Calibration of out-of-fold PIT (should be uniform)", "",
              "| fit | n | KS p-value | 50% interval coverage | 90% interval coverage | PIT histogram (10 bins) |", "|---|---|---|---|---|---|"]
        fig, axes = plt.subplots(2, 5, figsize=(15, 5.5))
        axes = axes.ravel()
        ai = 0
        for k, v in cv.items():
            cols = [("", v["pit"])] if "pit" in v else [(c[4:], v[c]) for c in v.columns if c.startswith("pit_")]
            for tag, pit in cols:
                pit = pit.to_numpy()
                cal = calibration(pit)
                B.append(f"| {k}{' ' + tag if tag else ''} | {cal['n']} | {cal['ks_p']:.3f} | {cal['cover50']:.2f} | {cal['cover90']:.2f} | {cal['hist']} |")
                if ai < len(axes) and (k.startswith("V1/") or (k == 'V3/joint' and tag == "cod_cost")):
                    axes[ai].hist(pit, bins=10, range=(0, 1), color="#4c72b0")
                    axes[ai].axhline(len(pit) / 10, color="k", ls="--", lw=0.8)
                    axes[ai].set_title(f"{k}{' ' + tag if tag else ''}", fontsize=8)
                    ai += 1
        for a in axes[ai:]:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "pit_histograms.png", dpi=110)
        plt.close(fig)
        B += ["", "Figure: `outputs/T1/figures/pit_histograms.png`. Expected coverage if calibrated: 0.50 and 0.90.", ""]

        base_f = p["outputs"] / "baseline" / "attempt_scores.parquet"
        k = "V1/cod_cost"
        if k in cv and base_f.exists():
            base = pd.read_parquet(base_f)
            t1 = df[KEYS + ["t_ref_total"]].assign(pred_cod=cv[k].set_index("attempt_row")["pred_raw"].reindex(df.index).to_numpy())
            t1["pred_t_total"] = t1["pred_cod"] + t1["t_ref_total"]
            m = t1.merge(base[KEYS + ["score_pred_q50", "score_primary"]], on=KEYS).merge(df[KEYS + ["t_total"]], on=KEYS)
            if scores is not None:
                m = m.merge(scores[KEYS + ["score_primary"]].rename(columns={"score_primary": "t1"}), on=KEYS)
                B += ["## Comparison with the LightGBM quantile baseline (out-of-fold, same attempts)", "",
                      f"- MAE of the predicted median `t_total`: T1 {np.mean(np.abs(m['pred_t_total'] - m['t_total'])):.3f} s vs baseline "
                      f"{np.mean(np.abs(m['score_pred_q50'] - m['t_total'])):.3f} s (n = {len(m)}). T1's prediction is `cod_cost` prediction + the forty reference time.",
                      f"- Spearman of T1 `score_primary` with the baseline score: {m['t1'].corr(m['score_primary'], method='spearman'):.3f}", ""]

    if "V3/joint" in full_fits:
        d, S = full_fits["V3/joint"]
        corr_u = np.mean(np.einsum("sij,skj->sik", S["L_u"], S["L_u"]), axis=0)
        corr_e = np.mean(np.einsum("sij,skj->sik", S["L_e"], S["L_e"]), axis=0)
        icc = np.mean(S["sigma_u"] ** 2 / (S["sigma_u"] ** 2 + S["sigma_e"] ** 2), axis=0)
        names = [n.replace("reaccel_ratio", "reaccel").replace("entry_ratio", "entry").replace("brake_dist", "brake") for n in JOINT_DIMS]
        fmt = lambda M: pd.DataFrame(M, index=names, columns=names).round(2).to_string()  # noqa: E731
        B += ["## V3: correlation of player traits (posterior mean)", "", "```", fmt(corr_u), "```", "",
              "Residual (within-attempt) correlation:", "", "```", fmt(corr_e), "```", "",
              "Model ICC per dimension: " + ", ".join(f"{n} {v:.2f}" for n, v in zip(names, icc)),
              "", "With ~110 players and a 9-d player effect, off-diagonal correlations are weakly identified and shrink toward the LKJ prior; read the pattern, not the values.", ""]
        pd.DataFrame(corr_u, index=JOINT_DIMS, columns=JOINT_DIMS).to_csv(out_dir / "v3_player_trait_corr.csv")

    # caterpillar of cod_cost player effects
    if "V1/cod_cost" in full_fits:
        d, S = full_fits["V1/cod_cost"]
        u, _ = player_effects(S, d)
        mean, lo, hi = u.mean(0) * d.stats["y_sd"], np.percentile(u, 5, 0) * d.stats["y_sd"], np.percentile(u, 95, 0) * d.stats["y_sd"]
        order = np.argsort(mean)
        n_per = attempts_per_player(d)
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.hlines(np.arange(len(order)), lo[order], hi[order], color=np.where(n_per[order] > 1, "#c44e52", "#8c8c8c"), lw=1)
        ax.plot(mean[order], np.arange(len(order)), "k.", ms=3)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("player effect on cod_cost (s); red = 2+ attempts")
        ax.set_ylabel("players, sorted")
        fig.tight_layout()
        fig.savefig(fig_dir / "player_effects_cod_cost.png", dpi=110)
        plt.close(fig)
        B += ["Figure: `outputs/T1/figures/player_effects_cod_cost.png` (posterior mean and 90% interval of each player's effect).", ""]

    B += ["## Risks and notes", "",
          "- Few attempts per player: player effects shrink heavily (see the shrinkage table). That is the correct behavior, not a defect.",
          "- Restricted to attempts passing QC, so feature-extraction errors are limited but not eliminated.",
          "- Plant-level targets share one residual scale across plants and no attempt-level random effect, as in the spec; the two plants of an attempt are only linked through the player effect.",
          "- `score_primary` standardizes by the new-player predictive scale sqrt(sigma_player^2 + sigma_attempt^2), so it measures deviation from the expectation given body and forty profile, not from the player's own typical level (that is `score_attempt_dev_cod_cost`).", ""]
    (out_dir / "cohort_summary.json").write_text(json.dumps({k: v for k, v in coh.items()}, indent=1, default=float))
    write_report("T1", {"t1": cfg, "n_attempts": len(df)}, seed, len(df), df["nfl_id"].nunique(), runtime, "\n".join(B))
    print(f"wrote {out_dir}")


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--variants", nargs="+")
    args = ap.parse_args()
    cfg = load_yaml("t1.yaml")
    seed = pipeline_cfg()["seed"]
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    mcfg = cfg["fast_mcmc"] if args.fast else cfg["mcmc"]
    targets = args.targets or cfg["targets"]
    variants = args.variants or cfg["variants"]
    do_cv = cfg["cv"] and not args.no_cv

    df, folds, pos_labels = load_data(cfg["population"])
    cond_cols = model_columns(df)
    n_pos = len(pos_labels)
    print(f"T1: {len(df)} attempts, {df['nfl_id'].nunique()} players, population={cfg['population']}, cv={do_cv}")

    results = {"fits": {}, "cv": {}, "cohort": {}, "loo": {}, "diag": {}, "joint": {}}
    full_fits = {}
    with Timer() as tm:
        for variant in [v for v in variants if v in ("V1", "V2")]:
            for target in targets:
                if variant == "V2" and TARGETS[target]["level"] != "plant":
                    continue
                key = f"{variant}/{target}"
                print(f"fit {key}")
                d, cond, grouped, S, diag = fit_target(df, target, cond_cols, n_pos, variant, cfg, mcfg, seed)
                results["diag"][key] = diag
                results["cohort"][key] = cohort_summary(S, d, d.columns)
                results["loo"][key] = psis_loo(grouped, d, S)
                full_fits[key] = (d, S)
                if do_cv:
                    print(f"cv  {key}")
                    results["cv"][key] = fold_scores(df, target, cond_cols, n_pos, variant, cfg, mcfg, seed, rng)
        if "V3" in variants:
            print("fit V3/joint")
            d, cond, grouped, S, diag = fit_joint(df, cond_cols, n_pos, cfg, mcfg, seed)
            results["diag"]["V3/joint"] = diag
            full_fits["V3/joint"] = (d, S)
            if do_cv:
                print("cv  V3/joint")
                results["cv"]["V3/joint"] = joint_fold_scores(df, cond_cols, n_pos, cfg, mcfg, seed, rng)
    build_outputs(df, results, full_fits, cfg, seed, tm.seconds, pos_labels, do_cv, args.fast)


if __name__ == "__main__":
    main()
