# %% [md]
# ### 8.1 Tuning the points table
#
# **Stage 1 (point values and season weights).** Tiers of 12 (the best strict size), remaining games against adjacent
# tiers, and a full factorial over level points per tier $w_{\text{level}} \in \{0, 2, 4, 6, 8, 12\}$, place points per
# position $w_{\text{place}} \in \{0, 0.5, 1\}$ and four season-weight profiles: 72 settings, each run for the full
# **50 seasons** on the tuning trajectories, next to the status quo and the best strict setting (tiers of 12, two up and
# two down) on the same futures.
#
# **Stage 2 (refinement).** Around the best stage-1 setting: tier sizes 8, 10, 12 and 16, and level and place values
# between the stage-1 grid points, on three trajectories.
#
# **Confirmation.** The best points setting, the best strict setting and the status quo on the six main-experiment
# trajectories (50 seasons each).

# %%
NAMES["fbs_points"] = "FBS-wide tiers (points)"
PALETTE["fbs_points"] = "#8a4fd0"
STRICT_BEST = ("fbs_tiers", dict(BEST["fbs_tiers"][1]))
SEASONS_PTS = RUN["seasons"]


def points_index(raw):
    """Mean index vs the status quo (both optimized / both 2019-style), mean movement, per candidate."""
    sq = raw[raw.cand == "sq"].set_index(["rep", "season"])[["optimized", "style2019"]]
    d = raw[raw.cand != "sq"].join(sq.rename(columns=lambda c: "sq_" + c), on=["rep", "season"])
    d["index_opt"] = 100 * d.optimized / d.sq_optimized
    d["index_2019tv"] = 100 * d.style2019 / d.sq_style2019
    g = d.groupby("cand", sort=False)
    out = g.agg(index_opt=("index_opt", "mean"), index_2019tv=("index_2019tv", "mean"),
                moved=("moved", lambda x: x.iloc[1:].mean() if len(x) > 1 else np.nan),
                never_on_tv=("never_on_tv", "mean"), marquee=("marquee_games", "mean"), min_games=("min_games", "min"))
    out["se"] = g.apply(lambda x: (x.index_opt.groupby(x.rep).mean()).std(ddof=1) / np.sqrt(x.rep.nunique())
                        if x.rep.nunique() > 1 else np.nan)
    return out.reset_index()


def cand_jobs(traj, reps, cands, tag):
    jobs = []
    for r in reps:
        jobs.append(dict(rep=r, traj=traj[r], policies={"sq": ("sq", {})}, sim=dict(free=False), tag=dict(cand="sq")))
        for name, (cls, prm) in cands.items():
            jobs.append(dict(rep=r, traj=traj[r], policies={cls: (cls, prm)}, sim=dict(free=False), tag=dict(cand=name)))
    return jobs


def pts_name(prm):
    return f"level {prm['level']:g} | place {prm['place']:g} | {prm['weights'].split(' (')[0]} | n {prm['n']}"


# ---- Stage 1
t0 = time.time()
pts_traj = simulate_power(RHO_INIT, LONG_RUN, SEASONS_PTS, max(3, RUN["doe_reps"] + 1), seed=SEED + 1)
LEVELS, PLACES = [0, 2, 4, 6, 8, 12], [0, 0.5, 1]
if RUN_MODE == "quick":
    LEVELS, PLACES = [0, 4, 12], [0.5]
cands1 = {"strict: " + json.dumps(STRICT_BEST[1]): STRICT_BEST}
for lv in LEVELS:
    for pl in PLACES:
        for wp in WEIGHT_PROFILES:
            prm = dict(n=12, level=lv, place=pl, weights=wp, fill="random")
            cands1[pts_name(prm)] = ("fbs_points", prm)
raw1, _ = run_jobs(cand_jobs(pts_traj, range(RUN["doe_reps"]), cands1, "s1"), "points_stage1")
S1 = points_index(raw1)
S1["kind"] = np.where(S1.cand.str.startswith("strict"), "strict", "points")
prm1 = {k: v[1] for k, v in cands1.items()}
for col in ("level", "place", "weights", "n"):
    S1[col] = S1.cand.map(lambda c: prm1[c].get(col, np.nan) if c in prm1 else np.nan)
S1.to_csv(OUT_DIR / "points_stage1.csv", index=False)
strict_idx = S1.loc[S1.kind == "strict", "index_opt"].iloc[0]
top1 = S1[S1.kind == "points"].sort_values("index_opt", ascending=False)
print(f"Stage 1: {len(cands1)} settings × {RUN['doe_reps']} trajectories × {SEASONS_PTS} seasons in {time.time() - t0:.0f}s")
print(f"Best strict setting (tiers of 12, 2 up / 2 down): index {strict_idx:.1f}\n")
print("Top 10 points settings:")
print(top1.head(10)[["cand", "index_opt", "index_2019tv", "moved", "marquee", "never_on_tv"]].round(2).to_string(index=False))
print("\nMean index by each factor (averaged over the other two):")
for col in ("level", "place", "weights"):
    print(S1[S1.kind == "points"].groupby(col).index_opt.agg(["mean", "max"]).round(2).T.to_string(), "\n")

# %%
fig, axes = plt.subplots(1, len(PLACES), figsize=(4.2 * len(PLACES), 3.6), sharey=True)
axes = np.atleast_1d(axes)
pts1 = S1[S1.kind == "points"]
vmin, vmax = pts1.index_opt.min(), max(pts1.index_opt.max(), strict_idx)
for ax, pl in zip(axes, PLACES):
    m = pts1[pts1.place == pl].pivot(index="weights", columns="level", values="index_opt").loc[list(WEIGHT_PROFILES)]
    im = ax.imshow(m.to_numpy(), cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
    for (i, j), v in np.ndenumerate(m.to_numpy()):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                color="white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.55 else INK)
    ax.set_xticks(range(m.shape[1]), [f"{c:g}" for c in m.columns]); ax.grid(False)
    ax.set_yticks(range(m.shape[0]), [w.split(" (")[0] for w in m.index])
    ax.set_xlabel("level points per tier"); ax.set_title(f"place points per position = {pl:g}", fontsize=9.5)
fig.colorbar(im, ax=axes, shrink=0.8, label="viewership index (status quo = 100)")
fig.suptitle(f"Points table, stage 1 (tiers of 12, 50 seasons). Best strict setting: {strict_idx:.1f}", fontsize=10.5)
plt.savefig(OUT_DIR / "points_stage1.png", dpi=150, bbox_inches="tight"); plt.show()

# %%
# ---- Stage 2: refine around the best stage-1 setting
b1 = prm1[top1.iloc[0].cand]
lv_grid = sorted({max(0, b1["level"] + d) for d in (-1, 0, 1)})
pl_grid = sorted({max(0, b1["place"] + d) for d in (-0.25, 0, 0.25)})
cands2 = {"strict: " + json.dumps(STRICT_BEST[1]): STRICT_BEST}
for n in (8, 10, 12, 16):
    for lv in lv_grid:
        for pl in pl_grid:
            prm = dict(n=n, level=lv, place=pl, weights=b1["weights"], fill="random")
            cands2[pts_name(prm)] = ("fbs_points", prm)
if RUN_MODE == "quick":
    cands2 = dict(list(cands2.items())[:4])
t0 = time.time()
reps2 = range(max(3, RUN["doe_reps"] + 1))
raw2, _ = run_jobs(cand_jobs(pts_traj, reps2, cands2, "s2"), "points_stage2")
S2 = points_index(raw2)
prm2 = {k: v[1] for k, v in cands2.items()}
for col in ("level", "place", "n"):
    S2[col] = S2.cand.map(lambda c: prm2[c].get(col, np.nan))
S2.to_csv(OUT_DIR / "points_stage2.csv", index=False)
top2 = S2[~S2.cand.str.startswith("strict")].sort_values("index_opt", ascending=False)
strict2 = S2.loc[S2.cand.str.startswith("strict"), "index_opt"].iloc[0]
print(f"Stage 2: {len(cands2)} settings × {len(reps2)} trajectories × {SEASONS_PTS} seasons in {time.time() - t0:.0f}s")
print(f"Best strict setting on these trajectories: {strict2:.1f}\n")
print(top2.head(12)[["cand", "index_opt", "se", "index_2019tv", "moved", "marquee"]].round(2).to_string(index=False))
print("\nBest index by tier size:")
print(top2.groupby("n").index_opt.max().round(2).to_string())
BEST_POINTS = ("fbs_points", prm2[top2.iloc[0].cand])
print("\nSelected:", pts_name(BEST_POINTS[1]))

# %% [md]
# ### 8.2 Confirmation on the main-experiment futures

# %%
t0 = time.time()
CONF_POLS = {"sq": ("sq", {}), "fbs_tiers": STRICT_BEST, "fbs_points": BEST_POINTS}
main_traj_pts = simulate_power(RHO_INIT, LONG_RUN, RUN["seasons"], RUN["reps"], seed=SEED + 2)
CONF_P, CONF_B = run_jobs([dict(rep=r, traj=main_traj_pts[r], policies=CONF_POLS, sim=dict(free=False, detail=(r == 0)))
                           for r in range(RUN["reps"])], "points_confirm")
Wc = CONF_P.pivot_table(index=["rep", "season"], columns="policy", values=["style2019", "optimized"])
Wc.columns = [f"{a}:{b}" for a, b in Wc.columns]
Wc = Wc.join(CONF_B.set_index(["rep", "season"])[["A_rank"]])
rows = []
for k in CONF_POLS:
    sub_ = CONF_P[CONF_P.policy == k]
    r_ = Wc[f"optimized:{k}"] / Wc["optimized:sq"]
    rows.append({"policy": NAMES[k], "setting": sub_.setting.iloc[0], "2019-style TV (M)": Wc[f"style2019:{k}"].mean(),
                 "optimized TV (M)": Wc[f"optimized:{k}"].mean(), "vs SQ, optimized %": 100 * (r_.mean() - 1),
                 "vs SQ, 2019-style %": 100 * (Wc[f"style2019:{k}"].mean() / Wc["style2019:sq"].mean() - 1),
                 "vs A %": 100 * (Wc[f"optimized:{k}"].mean() / Wc.A_rank.mean() - 1),
                 "seasons beating SQ %": 100 * (r_ > 1).mean(), "teams moving / season": sub_.moved.iloc[1:].mean() if k != "sq" else 0,
                 "marquee games": sub_.marquee_games.mean(), "never on TV": sub_.never_on_tv.mean()})
CONFIRM = pd.DataFrame(rows)
CONFIRM.to_csv(OUT_DIR / "points_confirmation.csv", index=False)
print(f"Confirmation: {RUN['reps']} trajectories × {RUN['seasons']} seasons in {time.time() - t0:.0f}s\n")
print(CONFIRM.drop(columns="setting").round(1).T.to_string(header=False))
d_ = (Wc["optimized:fbs_points"] - Wc["optimized:fbs_tiers"]) / Wc["optimized:sq"] * 100
per_rep = d_.groupby(level="rep").mean()
print(f"\nPoints minus strict (index points): mean {d_.mean():+.2f}, by trajectory {per_rep.round(2).tolist()}, "
      f"points ahead in {100 * (d_ > 0).mean():.0f}% of seasons")


# Mobility statistics from trajectory 0
def mobility(k):
    R = CONF_P[(CONF_P.policy == k) & (CONF_P.rep == 0)].sort_values("season")
    G = np.array(R.groups.tolist())
    top = G == G.min(1, keepdims=True)
    runs, jumps = [], 0
    for t in range(G.shape[1]):
        r = 0
        for v in top[:, t]:
            if v:
                r += 1
            elif r:
                runs.append(r); r = 0
        if r:
            runs.append(r)
    jumps = (np.abs(np.diff(G, axis=0)) >= 2).sum() / (G.shape[0] - 1)
    return {"teams changing tier / season": (G[1:] != G[:-1]).sum(1).mean(), "teams jumping 2+ tiers / season": jumps,
            "programs reaching the top tier": int(top.any(0).sum()), "median seasons per top-tier stay": float(np.median(runs))}


MOBILITY = pd.DataFrame({NAMES[k]: mobility(k) for k in ("fbs_tiers", "fbs_points")})
print("\nMobility (trajectory 0):")
print(MOBILITY.round(1).to_string())

# %%
fig, ax = plt.subplots(figsize=(10, 4.2))
by = Wc.groupby(level="season").mean()
for k in CONF_POLS:
    ax.plot(by.index + 1, by[f"optimized:{k}"], color=PALETTE[k], lw=2, label=f"{NAMES[k]}, optimized TV")
ax.plot(by.index + 1, by.A_rank, color=INK2, ls="-.", lw=1.4, label="2019 televised games by rank (A)")
ax.set_xlabel("Season"); ax.set_ylabel("Season viewers (millions)"); ax.set_ylim(0, None); ax.legend(fontsize=8.5, loc="lower right")
ax.set_title(f"Points table vs strict promotion/relegation, mean of {RUN['reps']} trajectories", fontsize=10, loc="left")
plt.savefig(OUT_DIR / "points_vs_strict_by_season.png", dpi=150, bbox_inches="tight"); plt.show()
