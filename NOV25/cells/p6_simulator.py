# %% [md]
# ## 7. Season simulator and the decomposition ladder
#
# One simulated season:
# 1. **Ratings.** True strengths come from the pre-generated trajectory (Section 5) and play the role of the preseason
#    rating in the viewership model (the information networks have when they set the TV schedule).
# 2. **Schedule.** The policy builds the 12-game schedule from last season's results (Section 6).
# 3. **Games.** Point margin = $\rho_i-\rho_j+15.5\,z_{ij}$, $z_{ij}\sim N(0,1)$, i.e. $P(i\text{ wins})=\Phi((\rho_i-\rho_j)/15.5)$.
# 4. **Television.** The 303 seats are filled from the season's games in two ways: the **2019-style rule** (Section 3.3,
#    no optimization) and the **optimizer** (`subset` mode: choose which games to televise and where, all primary
#    constraints including the rest rule).
# 5. **Update.** Standings and the results-based rating are computed and tiers/divisions are updated.
#
# **The ladder.** Each season also records benchmarks that do not depend on the policy:
#
# | Rung | Matchups | Television | What it isolates |
# |---|---|---|---|
# | **A. 2019 slate by rank** | the 303 games televised in 2019, by preseason rank (2019's #1 v #32 → this season's #1 v #32) | the 2019 week, day, window and medium of each game | the status quo as practiced (your requested baseline) |
# | A′. 2019 slate by team | the same 303 games between the same programs | as in 2019 | reference: the same programs, regardless of how good they are now |
# | **B. 2019 slate, re-slotted** | as A | optimal within each game's week (exact) | value of better slotting alone |
# | **D′. 2019 full slate, optimal TV** | all 723 FBS games of 2019, by rank, each in its 2019 week | optimizer chooses the 303 and their slots (exact) | value of choosing *which* games to show, with the 2019 schedule |
# | **Dw. 2019 full slate, free weeks** | as D′, but the week of each game is free | optimizer (as in D) | value of setting the calendar centrally |
# | **C. Status quo, 2019-style TV** | simulated status-quo schedule (2025 conferences) | 2019-style rule | simulated status quo with networks' 2019 habits (compare with A) |
# | **D. Status quo, optimal TV** | simulated status-quo schedule | optimizer | the prototype's "status quo" |
# | **E₀. Policy, 2019-style TV** | policy schedule | 2019-style rule | policy effect without optimization |
# | **E. Policy, optimal TV** | policy schedule | optimizer | policy with optimization |
# | **F. Theoretical maximum** | any pair may meet | optimizer | upper benchmark |
#
# The optimized rungs share the same window caps: σ × the window power of rung A this season. B and D′ need one exact MIP
# per week and are computed every `BENCH_EVERY`-th season on the first `BENCH_REPS` replications.
#
# **Common random numbers.** Within a replication every policy faces the same strength trajectory, the same warm-up
# season and the same game-level noise $z_{ij}$ for any matchup it schedules.

# %%
import os
import multiprocessing as mp

N_JOBS = min(os.cpu_count() or 1, 8)
BENCH_EVERY = 5
BENCH_REPS = {"quick": 1, "standard": 2, "paper": 5}[RUN_MODE]
TV_RATING = "current"          # "current": preseason rating = this season's strength; "previous": last season's (Section 10)


def make_scheduler(mode="subset", brand_mode="team", min_exposure=0, sigma=None, rest_rule=True, teams=None, conf=None, **solver):
    sigma = SIM["power_cap_slack"] if sigma is None else (None if sigma == "none" else sigma)
    cons = scenario_constraints(None, matchup_mode=mode, brand_mode=brand_mode, min_exposure=min_exposure,
                                power_cap_slack=sigma, rest_rule=rest_rule)
    return SeasonScheduler(teams or TEAMS_SIM, conf or CONF_SIM, cons, solver=dict(grasp_starts=RUN["grasp_starts"], **solver))


TID_SIM = {t: k for k, t in enumerate(TEAMS_SIM)}
TV_TEAM_PAIRS_2019 = np.sort(np.array([[TID_SIM[TEAMS_2019[a]], TID_SIM[TEAMS_2019[b]]] for a, b in PAIRS_2019]), axis=1)


def rank_slates(rho):
    """The 2019 televised slate and full FBS slate mapped to this season's teams by preseason rank."""
    team_at_rank = np.argsort(-rho, kind="stable")
    return np.sort(team_at_rank[TV_RANK_PAIRS_2019], axis=1), np.sort(team_at_rank[ALL_RANK_PAIRS_2019], axis=1)


def reference_power(rho, tv_pairs, how=None):
    how = how or SIM["cap_reference"]
    if how == "absolute_2019":
        return INV.Ptot.copy()
    return np.bincount(INV.pos_win[HIST_POS], weights=rho[tv_pairs].sum(1), minlength=len(INV.Ptot))


def score_fixed(sch, rho, pairs, positions, ref):
    """Realized viewership of games placed in given positions (no optimization)."""
    saved = sch.c["matchup_mode"]; sch.c["matchup_mode"] = "subset"
    sch._prepare(np.asarray(rho, float), np.asarray(pairs, int), None, ref_power=ref)
    sch.c["matchup_mode"] = saved
    return sch.evaluate(list(zip(range(len(pairs)), positions)))[1]


def tv_stats(res, rho):
    t = res.table
    on = np.zeros(len(rho), bool)
    if len(t):
        on[[TID_SIM[x] for x in t.team1]] = True; on[[TID_SIM[x] for x in t.team2]] = True
    return dict(tv_games=len(t), never_on_tv=int((~on).sum()), mean_tv_power=t.game_power.mean() if len(t) else np.nan,
                marquee_games=int((t.game_power >= 140).sum()) if len(t) else 0,
                broadcast_power=t.loc[t.medium == "broadcast", "game_power"].mean() if len(t) else np.nan)


class Schedulers:
    """The schedulers used in one experiment arm (built once per process)."""

    def __init__(self, sigma=None, brand_mode="team", min_exposure=0):
        kw = dict(sigma=sigma, brand_mode=brand_mode)
        self.opt = make_scheduler("subset", min_exposure=min_exposure, **kw)
        self.free = make_scheduler("free", min_exposure=min_exposure, **kw)
        self.pattern = make_scheduler("subset", **kw)
        self.fixed = make_scheduler("fixed_week", rest_rule=False, **kw)
        self.subweek = make_scheduler("subset_week", rest_rule=False, **kw)
        self.score = make_scheduler("subset", **kw)


def simulate_rep(rep, traj, policies, scheds, seasons=None, bench=False, free=True, pattern=True, cap_reference=None,
                 tv_rating=None, seed=SEED, detail=False):
    """One replication. policies: {key: Policy instance} (include "sq"). Returns (policy rows, benchmark rows)."""
    tv_rating = tv_rating or TV_RATING
    S = traj.shape[0] if seasons is None else seasons
    wp, wm = warmup_season(traj[0], rep, seed)
    for P in policies.values():
        P.setup(wp, wm)
    prow, brow = [], []
    for s in range(S):
        rho = traj[s]
        rho_tv = rho if (tv_rating == "current" or s == 0) else traj[s - 1]
        tv_pairs, all_pairs = rank_slates(rho_tv)
        ref = reference_power(rho_tv, tv_pairs, cap_reference)
        b = dict(rep=rep, season=s)
        b["A_rank"] = score_fixed(scheds.score, rho_tv, tv_pairs, HIST_POS, ref)
        b["A_team"] = score_fixed(scheds.score, rho_tv, TV_TEAM_PAIRS_2019, HIST_POS, ref)
        if bench and s % BENCH_EVERY == 0:
            b["B_rank"] = scheds.fixed.solve(rho_tv, pairs=tv_pairs, pair_week=WEEK_2019, ref_power=ref,
                                             start=list(zip(range(len(tv_pairs)), HIST_POS))).realized
            b["Dp_rank"] = scheds.subweek.solve(rho_tv, pairs=all_pairs, pair_week=ALL_WEEK_2019, ref_power=ref).realized
        b["Dw_rank"] = scheds.opt.solve(rho_tv, pairs=all_pairs, ref_power=ref, seed=seed + s).realized
        if free:
            b["F_max"] = scheds.free.solve(rho_tv, ref_power=ref, seed=seed + s).realized
        brow.append(b)
        z = game_noise(rep, s, seed=seed)
        for key, P in policies.items():
            rng = np.random.default_rng([seed, rep, s, zlib.crc32(key.encode()) % 10_000])
            pairs = P.schedule(rng)
            margin = rho[pairs[:, 0]] - rho[pairs[:, 1]] + SIGMA_GAME * z[pairs[:, 0], pairs[:, 1]]
            opt = scheds.opt.solve(rho_tv, pairs=pairs, ref_power=ref, seed=int(rng.integers(1 << 30)))
            row = dict(rep=rep, season=s, policy=key, setting=P.label(), optimized=opt.realized,
                       linear=opt.linear, feasible=not opt.stats["violations"], sched_games=len(pairs),
                       min_games=int(np.bincount(pairs.ravel(), minlength=CTX.N).min()), **tv_stats(opt, rho_tv))
            if pattern:
                row["style2019"] = pattern_tv(scheds.pattern, rho_tv, pairs, SLOT_ORDER_2019, ref_power=ref).realized
            grp = pd.Series(rho).groupby(P.groups()).mean().sort_values(ascending=False)
            row.update({f"group{k + 1}_power": grp.iloc[k] for k in range(4)})
            if detail:                     # keep tiers and the televised games (for the rulebook statistics)
                row["groups"] = np.asarray(P.groups()).tolist()
                row["tv"] = [(TID_SIM[r.team1], TID_SIM[r.team2], r.medium, r.time, int(r.week), round(float(r.realized_view), 3))
                             for r in opt.table.itertuples()]
                row["rho"] = np.round(rho, 2).tolist()
            g_before = np.asarray(P.groups()).copy()
            prow.append(row)
            P.update(pairs, margin)
            row["moved"] = int((np.asarray(P.groups()) != g_before).sum())     # teams changing tier/division
            row.update(getattr(P, "last_stats", {}))
    return prow, brow


# ---- parallel execution over (arm, replication) jobs ----
_JOBS = []


def _run_job(k):
    spec = _JOBS[k]
    sch = Schedulers(**spec.get("sched", {}))
    pols = {key: POLICY_CLASSES[cls_key](CTX, **prm) for key, (cls_key, prm) in spec["policies"].items()}
    t0 = time.time()
    prow, brow = simulate_rep(spec["rep"], spec["traj"], pols, sch, **spec.get("sim", {}))
    for r in prow + brow:
        r.update(spec.get("tag", {}))
    return prow, brow, time.time() - t0


CACHE_DIR = OUT_DIR / "cache"          # finished stages are cached here; delete the folder to recompute everything


def run_jobs(jobs, name=None):
    """Run simulate_rep for each job spec; in parallel (fork) when N_JOBS > 1. Returns (policy df, benchmark df).
    With `name`, results are cached in outputs/cache/<name>_<RUN_MODE>.pkl and reused on a rerun."""
    global _JOBS
    if name:
        CACHE_DIR.mkdir(exist_ok=True)
        f = CACHE_DIR / f"{name}_{RUN_MODE}.pkl"
        if f.exists():
            print(f"(loaded cached results: {f.name})")
            return pd.read_pickle(f)
        out = run_jobs(jobs)
        pd.to_pickle(out, f)
        return out
    _JOBS = jobs
    if N_JOBS > 1 and len(jobs) > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(N_JOBS) as pool:
            out = pool.map(_run_job, range(len(jobs)), chunksize=1)
    else:
        out = [_run_job(k) for k in range(len(jobs))]
    prow = [r for o in out for r in o[0]]
    brow = [r for o in out for r in o[1]]
    return pd.DataFrame(prow), pd.DataFrame(brow)


# Timing check: one season of every rung
_traj = simulate_power(RHO_INIT, LONG_RUN, 1, 1, seed=SEED)[0]
t0 = time.time()
_p, _b = simulate_rep(0, _traj, {"sq": StatusQuo(CTX), "fbs_tiers": FBSTiers(CTX)}, Schedulers(), bench=True)
print(f"One season, all rungs: {time.time() - t0:.1f}s  (N_JOBS={N_JOBS})")
print(pd.DataFrame(_b).round(1).to_string(index=False))
print(pd.DataFrame(_p)[["policy", "style2019", "optimized", "never_on_tv", "mean_tv_power", "marquee_games", "min_games", "feasible"]]
      .round(1).to_string(index=False))
