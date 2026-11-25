# %% [md]
# ### 8.3 More fluid settings from the plateau
# Two settings from the stage-1 plateau (within 0.5 index points of the best) are confirmed on the six main
# trajectories: the one with the median amount of movement (fluid A) and the one that moves the most teams (fluid B).

# %%
pl1 = S1[S1.kind == "points"]
plateau = pl1[pl1.index_opt >= pl1.index_opt.max() - 0.5]                 # within 0.5 index points of the stage-1 best
most = plateau.sort_values("moved", ascending=False).iloc[0]
mid = plateau.iloc[(plateau.moved - plateau.moved.median()).abs().argsort()].iloc[0]
FLUID = {"fluid A": ("fbs_points", prm1[mid.cand]), "fluid B": ("fbs_points", prm1[most.cand])}
print(f"Stage-1 plateau (within 0.5 of {pl1.index_opt.max():.2f}): {len(plateau)} settings; "
      f"median-movement pick: {mid.cand}; most-movement pick: {most.cand}")
jobs = []
for r in range(RUN["reps"]):
    for name, spec in FLUID.items():
        jobs.append(dict(rep=r, traj=main_traj_pts[r], policies={"fbs_points": spec}, sim=dict(free=False, detail=(r == 0)),
                         tag=dict(cand=name)))
FL_P, _ = run_jobs(jobs, "points_fluid")
rows = []
for name in FLUID:
    s_ = FL_P[FL_P.cand == name].set_index(["rep", "season"])
    r_ = s_.optimized / Wc["optimized:sq"]
    R0 = FL_P[(FL_P.cand == name) & (FL_P.rep == 0)].sort_values("season")
    G = np.array(R0.groups.tolist()); top = G == G.min(1, keepdims=True)
    rows.append({"setting": pts_name(FLUID[name][1]), "optimized TV (M)": s_.optimized.mean(),
                 "vs SQ, optimized %": 100 * (r_.mean() - 1), "vs SQ, 2019-style %": 100 * (s_.style2019.mean() / Wc["style2019:sq"].mean() - 1),
                 "vs strict, index pts": 100 * ((s_.optimized - Wc["optimized:fbs_tiers"]) / Wc["optimized:sq"]).mean(),
                 "seasons beating SQ %": 100 * (r_ > 1).mean(), "teams moving / season": s_.moved.groupby(level="rep").apply(lambda x: x.iloc[1:].mean()).mean(),
                 "jump 2+ tiers / season": (np.abs(np.diff(G, axis=0)) >= 2).sum() / (G.shape[0] - 1),
                 "programs reaching top tier": int(top.any(0).sum()), "marquee games": s_.marquee_games.mean(), "never on TV": s_.never_on_tv.mean()})
FLUID_TAB = pd.DataFrame(rows)
FLUID_TAB.to_csv(OUT_DIR / "points_fluid_confirmation.csv", index=False)
print(FLUID_TAB.round(2).T.to_string(header=False))
