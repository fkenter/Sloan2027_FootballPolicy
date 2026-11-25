# %% [md]
# ### 8.4 A bonus for winning your tier
# Fluid B plus a bonus for finishing first in one's tier, swept over 0 to 36 points (one win = 1 point) on the six main
# trajectories, 50 seasons each, next to the status quo. The change against no bonus is a paired difference (same
# futures, same game noise); its standard error is taken across the six trajectories.

# %%
CHAMP_BASE = dict(FLUID["fluid B"][1])
BONUSES = [0, 3, 6, 9, 12, 18, 24, 36]
jobs = []
for r in range(RUN["reps"]):
    jobs.append(dict(rep=r, traj=main_traj_pts[r], policies={"sq": ("sq", {})}, sim=dict(free=False), tag=dict(bonus=-1)))
    for b in BONUSES:
        jobs.append(dict(rep=r, traj=main_traj_pts[r], policies={"fbs_champ": ("fbs_champ", {**CHAMP_BASE, "champ": float(b)})},
                         sim=dict(free=False), tag=dict(bonus=b)))
t0 = time.time()
CH, _ = run_jobs(jobs, "champ_bonus")
sqc = CH[CH.bonus == -1].set_index(["rep", "season"])[["optimized", "style2019"]]
dc = CH[CH.bonus >= 0].join(sqc.rename(columns=lambda c: "sq_" + c), on=["rep", "season"])
dc["idx"] = 100 * dc.optimized / dc.sq_optimized
dc["idx19"] = 100 * dc.style2019 / dc.sq_style2019
base0 = dc[dc.bonus == 0].set_index(["rep", "season"]).idx
dc["diff0"] = dc.idx.to_numpy() - base0.reindex(pd.MultiIndex.from_frame(dc[["rep", "season"]])).to_numpy()
rows = []
for b, g in dc.groupby("bonus"):
    per_rep = g.groupby("rep").diff0.mean()
    later = g[g.season > 0]
    rows.append(dict(bonus=b, viewers_M=g.optimized.mean(), index_opt=g.idx.mean(), index_2019tv=g.idx19.mean(),
                     change_vs_no_bonus=g.diff0.mean(), se=per_rep.std(ddof=1) / np.sqrt(len(per_rep)),
                     futures_better_than_0=int((per_rep > 0).sum()), teams_moving=later.moved.mean(),
                     winners_promoted_pct=100 * later.winner_promoted.mean(), winners_jump2_pct=100 * later.winner_jump2.mean(),
                     marquee=g.marquee_games.mean()))
CHAMP = pd.DataFrame(rows)
CHAMP.to_csv(OUT_DIR / "tier_winner_bonus_sweep.csv", index=False)
print(f"Tier-winner bonus on {pts_name(CHAMP_BASE)}: {len(jobs)} runs in {time.time() - t0:.0f}s; "
      f"status quo (optimized TV) {sqc.optimized.mean():.1f}M")
print(CHAMP.round(2).to_string(index=False))
