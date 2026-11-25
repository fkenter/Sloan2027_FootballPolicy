# %% [md]
# ## 10. Robustness
# Each check reruns the status quo and the tuned policies on the first `rob_reps` replications of the main experiment:
# * **Power cap σ = 1.10 and no cap**: how much of the result is driven by the per-window quality cap (A.8).
# * **Absolute 2019 cap**: window caps fixed at the 2019 window totals instead of the rank-equivalent values.
# * **Minimum exposure = 1** (A.13): every team televised at least once a season.
# * **Brand-neutral viewership** (only with `WITH_CONFERENCES`): every team gets the league-average brand effect.
# * **Lagged ratings**: networks schedule with last season's rating (the viewership model's prior-year-end rating),
#   while games are played with this season's strength.

# %%
t_rob = time.time()
ARMS = {
    "baseline": dict(),
    "σ = 1.10": dict(sched=dict(sigma=1.10)),
    "no power cap": dict(sched=dict(sigma="none")),
    "absolute 2019 cap": dict(sim=dict(cap_reference="absolute_2019")),
    "min exposure = 1": dict(sched=dict(min_exposure=1)),
    "lagged ratings": dict(sim=dict(tv_rating="previous")),
}
if WITH_CONFERENCES:
    ARMS["brand-neutral"] = dict(sched=dict(brand_mode="average"))
jobs = []
for arm, spec in ARMS.items():
    for r in range(RUN["rob_reps"]):
        jobs.append(dict(rep=r, traj=main_traj[r], policies=POLICIES, sched=spec.get("sched", {}),
                         sim=dict(spec.get("sim", {})), tag=dict(arm=arm)))
ROB_P, ROB_B = run_jobs(jobs, "robustness")
rob = []
for arm in ARMS:
    p = ROB_P[ROB_P.arm == arm].pivot_table(index=["rep", "season"], columns="policy", values=["style2019", "optimized"])
    p.columns = [f"{a}:{b}" for a, b in p.columns]
    p = p.join(ROB_B[ROB_B.arm == arm].set_index(["rep", "season"])[["A_rank", "F_max"]])
    row = {"check": arm, "A (2019 slate by rank)": p.A_rank.mean(), "SQ 2019-style TV": p["style2019:sq"].mean(),
           "SQ optimized TV": p["optimized:sq"].mean(), "max": p.F_max.mean(),
           "optimization effect (SQ) %": 100 * (p["optimized:sq"].mean() / p["style2019:sq"].mean() - 1)}
    for k in POL:
        row[f"{NAMES[k]}: policy effect, opt TV %"] = 100 * (p[f"optimized:{k}"].mean() / p["optimized:sq"].mean() - 1)
        row[f"{NAMES[k]}: policy effect, 2019-style TV %"] = 100 * (p[f"style2019:{k}"].mean() / p["style2019:sq"].mean() - 1)
    row["never on TV (SQ, opt)"] = ROB_P[(ROB_P.arm == arm) & (ROB_P.policy == "sq")].never_on_tv.mean()
    rob.append(row)
ROB = pd.DataFrame(rob).set_index("check")
ROB.to_csv(OUT_DIR / "robustness.csv")
print(f"Robustness checks ({RUN['rob_reps']} replications × {RUN['seasons']} seasons each) in {time.time() - t_rob:.0f}s\n")
print(ROB.round(1).T.to_string())
