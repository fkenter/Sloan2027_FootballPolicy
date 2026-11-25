# %% [md]
# ## 8. Experimental design: tuning each policy
#
# For each policy, a full factorial over its settings. Every candidate is simulated on the same `doe_reps` strength
# trajectories (common random numbers) for `doe_seasons` seasons, next to the status quo. The response is the
# **viewership index**: the policy's season viewership with optimized television divided by the status quo's with
# optimized television in the same replication and season (status quo = 100), averaged over seasons and replications.
# The same index with 2019-style television is reported alongside.
#
# | Policy | Factors |
# |---|---|
# | Like-opponent games | `k` ∈ {1, 2, 3, 4} like-opponent games per team |
# | Conference tiers | `c` ∈ {1, 2, 3, 4} crossover games × `m` ∈ {1, 2, 3, re-sort} teams swapped per division |
# | FBS-wide tiers | `n` ∈ {6, 8, 10, 12} teams per tier × `p` ∈ {1, 2, 3, re-sort} promoted/relegated × `fill` ∈ {adjacent, random} |
#
# Settings within one index point of the best are listed as practical alternatives.

# %%
PALETTE = {"sq": "#2a78d6", "like": "#eb6834", "conf_tiers": "#1baf7a", "fbs_tiers": "#eda100"}
NAMES = {"sq": "Status quo", "like": "Like-opponent games", "conf_tiers": "Conference tiers", "fbs_tiers": "FBS-wide tiers"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "font.size": 10, "legend.frameon": False, "axes.axisbelow": True})

GRID_DOE = {
    "like": [dict(k=k) for k in (1, 2, 3, 4)],
    "conf_tiers": [dict(c=c, m=m) for c in (1, 2, 3, 4) for m in (1, 2, 3, "re-sort")],
    "fbs_tiers": [dict(n=n, p=p, fill=f) for n in (6, 8, 10, 12) for p in (1, 2, 3, "re-sort") for f in ("adjacent", "random")],
}
if RUN_MODE == "quick":
    GRID_DOE = {"like": [dict(k=2), dict(k=3)], "conf_tiers": [dict(c=3, m=2), dict(c=2, m="re-sort")],
                "fbs_tiers": [dict(n=8, p=2, fill="adjacent"), dict(n=12, p=2, fill="random"), dict(n=8, p="re-sort", fill="adjacent")]}

t_doe = time.time()
doe_traj = simulate_power(RHO_INIT, LONG_RUN, RUN["doe_seasons"], RUN["doe_reps"], seed=SEED + 1)
jobs = []
for r in range(RUN["doe_reps"]):
    jobs.append(dict(rep=r, traj=doe_traj[r], policies={"sq": ("sq", {})}, sim=dict(free=False), tag=dict(cand="sq")))
    for key, grid in GRID_DOE.items():
        for prm in grid:
            cand = key + "|" + json.dumps(prm)
            jobs.append(dict(rep=r, traj=doe_traj[r], policies={key: (key, prm)}, sim=dict(free=False), tag=dict(cand=cand)))
DOE_RAW, _ = run_jobs(jobs, "doe")
sq_doe = DOE_RAW[DOE_RAW.cand == "sq"].set_index(["rep", "season"])
DOE_RAW = DOE_RAW.join(sq_doe[["optimized", "style2019"]].rename(columns=lambda c: "sq_" + c), on=["rep", "season"])
DOE_RAW["index_opt"] = 100 * DOE_RAW.optimized / DOE_RAW.sq_optimized
DOE_RAW["index_2019tv"] = 100 * DOE_RAW.style2019 / DOE_RAW.sq_style2019
DOE = (DOE_RAW[DOE_RAW.cand != "sq"].groupby(["policy", "cand"], sort=False)
       .agg(setting=("setting", "first"), index_opt=("index_opt", "mean"), index_2019tv=("index_2019tv", "mean"),
            never_on_tv=("never_on_tv", "mean"), min_games=("min_games", "min")).reset_index())
DOE.to_csv(OUT_DIR / "doe_results.csv", index=False)
print(f"DOE: {DOE.cand.nunique()} settings × {RUN['doe_reps']} reps × {RUN['doe_seasons']} seasons in {time.time() - t_doe:.0f}s\n")

BEST = {"sq": ("sq", {})}
for key in GRID_DOE:
    sub = DOE[DOE.policy == key].sort_values("index_opt", ascending=False)
    BEST[key] = (key, json.loads(sub.iloc[0].cand.split("|", 1)[1]))
    near = sub[sub.index_opt >= sub.iloc[0].index_opt - 1]
    print(f"{NAMES[key]}: best {sub.iloc[0].setting} → index {sub.iloc[0].index_opt:.1f} (optimized TV), "
          f"{sub.iloc[0].index_2019tv:.1f} (2019-style TV); {len(near)} of {len(sub)} settings within 1 point")
    print(sub.head(8)[["setting", "index_opt", "index_2019tv", "never_on_tv", "min_games"]].round(2).to_string(index=False), "\n")

# %%
fb = DOE[DOE.policy == "fbs_tiers"].copy()
if len(fb) > 3:
    prm = fb.cand.str.split("|", n=1).str[1].map(json.loads)
    fb["n"], fb["p"], fb["fill"] = prm.map(lambda d: d["n"]), prm.map(lambda d: str(d["p"])), prm.map(lambda d: d["fill"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fill in zip(axes, ("adjacent", "random")):
        m = fb[fb.fill == fill].pivot(index="p", columns="n", values="index_opt").loc[["1", "2", "3", "re-sort"]]
        im = ax.imshow(m.to_numpy(), cmap="Blues", aspect="auto", vmin=fb.index_opt.min() - 1, vmax=fb.index_opt.max())
        for (i, j), v in np.ndenumerate(m.to_numpy()):
            frac = (v - (fb.index_opt.min() - 1)) / (fb.index_opt.max() - fb.index_opt.min() + 1)
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="white" if frac > 0.55 else INK)
        ax.set_xticks(range(m.shape[1]), m.columns); ax.set_yticks(range(m.shape[0]), m.index); ax.grid(False)
        ax.set_xlabel("teams per tier (n)"); ax.set_title(f"fill = {fill}", fontsize=10)
    axes[0].set_ylabel("promoted / relegated per tier (p)")
    fig.colorbar(im, ax=axes, shrink=.8, label="viewership index (status quo = 100)")
    fig.suptitle("FBS-wide tiers: viewership index by setting (optimized television)", fontsize=11)
    plt.savefig(OUT_DIR / "doe_fbs_tiers.png", dpi=150, bbox_inches="tight"); plt.show()
