# %% [md]
# ## 9. Main experiment: where do the gains come from?
# The status quo and each policy at its best DOE setting, over `seasons` seasons × `reps` replications, with every rung of
# the ladder (Section 7). B and D′ (exact MIPs) are computed every 5th season on the first `BENCH_REPS` replications.

# %%
t_main = time.time()
main_traj = simulate_power(RHO_INIT, LONG_RUN, RUN["seasons"], RUN["reps"], seed=SEED + 2)
POLICIES = dict(BEST)
MAIN, BENCH = run_jobs([dict(rep=r, traj=main_traj[r], policies=POLICIES, sim=dict(bench=r < BENCH_REPS, detail=(r == 0)))
                        for r in range(RUN["reps"])], "main")
print(f"Main experiment: {len(MAIN)} policy-seasons, {len(BENCH)} benchmark seasons in {time.time() - t_main:.0f}s; "
      f"all schedules feasible: {MAIN.feasible.all()}; every team 12 games: {(MAIN.min_games == SEASON_GAMES).all()}")
MAIN.drop(columns=[c for c in ("groups", "tv", "rho") if c in MAIN]).to_csv(OUT_DIR / "main_experiment_seasons.csv", index=False)
BENCH.to_csv(OUT_DIR / "main_benchmarks_seasons.csv", index=False)

W = MAIN.pivot_table(index=["rep", "season"], columns="policy", values=["style2019", "optimized"])
W.columns = [f"{a}:{b}" for a, b in W.columns]
W = W.join(BENCH.set_index(["rep", "season"]))
POL = [k for k in POLICIES if k != "sq"]

# %% [md]
# ### 9.1 The 2 × 2 decomposition (all seasons and replications)
# Rows: who plays whom. Columns: how the 303 televised games are chosen and slotted. Mean season viewers (millions).
# *Optimization effect* = moving right within a row; *policy effect* = moving down within a column.

# %%
rows = []
for k in POL:
    C, D = W["style2019:sq"], W["optimized:sq"]
    E0, E = W[f"style2019:{k}"], W[f"optimized:{k}"]
    rows.append({"policy": NAMES[k], "setting": MAIN[MAIN.policy == k].setting.iloc[0],
                 "SQ, 2019-style TV": C.mean(), "SQ, optimized TV": D.mean(),
                 "policy, 2019-style TV": E0.mean(), "policy, optimized TV": E.mean(),
                 "optimization effect (SQ) %": 100 * (D.mean() / C.mean() - 1),
                 "policy effect, 2019-style TV %": 100 * (E0.mean() / C.mean() - 1),
                 "policy effect, optimized TV %": 100 * (E.mean() / D.mean() - 1),
                 "both vs SQ 2019-style %": 100 * (E.mean() / C.mean() - 1),
                 "both vs 2019 slate by rank (A) %": 100 * (E.mean() / W.A_rank.mean() - 1),
                 "policy wins seasons % (opt TV)": 100 * (E > D).mean()})
TWO_BY_TWO = pd.DataFrame(rows)
TWO_BY_TWO.to_csv(OUT_DIR / "decomposition_2x2.csv", index=False)
print(TWO_BY_TWO.drop(columns="setting").round(1).T.to_string(header=False))

# %% [md]
# ### 9.2 The ladder: from the 2019 slate to the theoretical maximum
# Computed on the seasons where every rung is available (B and D′ seasons), so each step is a like-for-like ratio.
# Each step's share of the total log gain from A to the best policy is shown.

# %%
sub = W.dropna(subset=["B_rank", "Dp_rank"])
best_pol = max(POL, key=lambda k: W[f"optimized:{k}"].mean())
ladder = [("A. 2019 televised games by rank, 2019 slots", "A_rank"),
          ("A′. same games by program (reference)", "A_team"),
          ("B. A re-slotted optimally within each week", "B_rank"),
          ("D′. 2019 full schedule by rank, optimal choice of TV games", "Dp_rank"),
          ("Dw. same, with the calendar (weeks) also free", "Dw_rank"),
          ("C. simulated status quo, 2019-style TV", "style2019:sq"),
          ("D. simulated status quo, optimized TV", "optimized:sq")]
ladder += [(f"E₀. {NAMES[k]}, 2019-style TV", f"style2019:{k}") for k in POL]
ladder += [(f"E. {NAMES[k]}, optimized TV", f"optimized:{k}") for k in POL]
ladder += [("F. theoretical maximum (any pair may meet)", "F_max")]
LADDER = pd.DataFrame([{"rung": lab, "col": c, "mean_viewers_M": sub[c].mean(), "index_A=100": 100 * sub[c].mean() / sub.A_rank.mean(),
                        "index_D=100": 100 * sub[c].mean() / sub["optimized:sq"].mean()} for lab, c in ladder])
LADDER.to_csv(OUT_DIR / "decomposition_ladder.csv", index=False)
print(f"Ladder on {len(sub)} season-replications with every rung:\n")
print(LADDER.drop(columns="col").round(1).to_string(index=False))

steps = [("Re-slot the 2019 games within their weeks (A→B)", "A_rank", "B_rank"),
         ("Choose which 2019 games to televise (B→D′)", "B_rank", "Dp_rank"),
         ("Set the calendar centrally: weeks free (D′→Dw)", "Dp_rank", "Dw_rank"),
         ("2019 schedule → simulated status-quo schedule, 2025 conferences (Dw→D)", "Dw_rank", "optimized:sq"),
         (f"Policy: {NAMES[best_pol]} (D→E)", "optimized:sq", f"optimized:{best_pol}"),
         ("Remaining to the theoretical maximum (E→F)", f"optimized:{best_pol}", "F_max")]
tot = np.log(sub[f"optimized:{best_pol}"].mean() / sub.A_rank.mean())
PATH = pd.DataFrame([{"step": s, "gain %": 100 * (sub[b].mean() / sub[a].mean() - 1),
                      "share of A→E log gain %": 100 * np.log(sub[b].mean() / sub[a].mean()) / tot if b != "F_max" else np.nan}
                     for s, a, b in steps])
PATH.to_csv(OUT_DIR / "decomposition_path.csv", index=False)
print(f"\nPath from the 2019 slate (A) to the best policy ({NAMES[best_pol]}, E): "
      f"+{100 * (np.exp(tot) - 1):.1f}% in total")
print(PATH.round(1).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(10, 0.42 * len(LADDER) + 1))
colors = []
for c in LADDER.col:
    k = c.split(":")[1] if ":" in c else None
    colors.append(PALETTE[k] if k else "#a3a29b")
y = np.arange(len(LADDER))[::-1]
alpha = [0.45 if c.startswith("style2019") else 1.0 for c in LADDER.col]
for yi, v, col, a in zip(y, LADDER.mean_viewers_M, colors, alpha):
    ax.barh(yi, v, color=col, alpha=a, height=0.62)
    ax.text(v + 8, yi, f"{v:,.0f}M", va="center", fontsize=9, color=INK)
ax.axvline(LADDER.mean_viewers_M.iloc[0], color=INK2, lw=1, ls="--")
ax.set_yticks(y, LADDER.rung, fontsize=9); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean regular-season viewers of the 303 televised games (millions)")
ax.set_xlim(0, LADDER.mean_viewers_M.max() * 1.12)
ax.set_title("From the 2019 slate to the theoretical maximum (pale bars: 2019-style TV, no optimization)", fontsize=11, loc="left")
plt.savefig(OUT_DIR / "decomposition_ladder.png", dpi=150, bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(figsize=(9, 4.8))
keys = list(POLICIES)
x = np.arange(len(keys))
for off, col, lab in [(-0.2, "style2019", "2019-style TV (networks' 2019 habits)"), (0.2, "optimized", "optimized TV")]:
    vals = [W[f"{col}:{k}"].mean() for k in keys]
    ax.bar(x + off, vals, width=0.38, color=[PALETTE[k] for k in keys], alpha=0.45 if col == "style2019" else 1.0,
           edgecolor="white", linewidth=2)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 10, f"{v:,.0f}", ha="center", fontsize=8.5, color=INK)
ax.axhline(W.A_rank.mean(), color=INK2, ls="--", lw=1)
ax.set_xlabel(f"Dashed line: the 2019 televised games by preseason rank, as slotted in 2019 (A) = {W.A_rank.mean():,.0f}M",
              fontsize=8.5, color=INK2)
ax.set_xticks(x, [NAMES[k] for k in keys]); ax.grid(axis="x", visible=False)
ax.set_ylabel("Mean season viewers (millions)")
ax.set_title("Pale: 2019-style TV · solid: optimized TV. Within a pair = optimization; across pairs = policy",
             fontsize=10, loc="left")
plt.savefig(OUT_DIR / "decomposition_2x2.png", dpi=150, bbox_inches="tight"); plt.show()

# %% [md]
# ### 9.3 Over time and season by season

# %%
fig, ax = plt.subplots(figsize=(11, 5))
by = W.groupby(level="season").mean()
ax.plot(by.index + 1, by.F_max, color=INK, ls=":", lw=2, label="Theoretical maximum")
for k in POLICIES:
    ax.plot(by.index + 1, by[f"optimized:{k}"], color=PALETTE[k], lw=2, label=f"{NAMES[k]}, optimized TV")
ax.plot(by.index + 1, by["style2019:sq"], color=PALETTE["sq"], lw=1.5, ls="--", label="Status quo, 2019-style TV")
ax.plot(by.index + 1, by.A_rank, color=INK2, lw=1.5, ls="-.", label="2019 televised games by rank, as slotted")
ax.set_xlabel("Season"); ax.set_ylabel("Season viewers (millions)"); ax.legend(ncol=2, fontsize=8.5, loc="lower right")
ax.set_ylim(0, None); ax.set_title(f"Mean over {RUN['reps']} replications", fontsize=10, loc="left")
plt.savefig(OUT_DIR / "viewership_by_season.png", dpi=150, bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(figsize=(9, 4.8))
data = [100 * W[f"optimized:{k}"] / W["optimized:sq"] for k in POL] + [100 * W.F_max / W["optimized:sq"]]
bp = ax.boxplot(data, patch_artist=True, tick_labels=[NAMES[k] for k in POL] + ["Theoretical\nmaximum"], widths=0.5,
                medianprops=dict(color=INK), flierprops=dict(ms=3, mec=INK2))
for patch, c in zip(bp["boxes"], [PALETTE[k] for k in POL] + ["#a3a29b"]):
    patch.set_facecolor(c); patch.set_alpha(.75)
ax.axhline(100, color=INK2, ls="--", lw=1)
ax.set_ylabel("Viewership index (status quo with optimized TV = 100)"); ax.grid(axis="x", visible=False)
ax.set_title("Policy effect season by season (both with optimized television)", fontsize=11, loc="left")
plt.savefig(OUT_DIR / "viewership_boxplot_normalized.png", dpi=150, bbox_inches="tight"); plt.show()

# Strength of the top four conferences (status quo) vs the top four FBS-wide tiers
fig, ax = plt.subplots(figsize=(11, 5.5))
ramp = ["#0d366b", "#2a78d6", "#79aef0", "#b9d5f7"]
for g in range(4):
    for key, ls, lab in [("sq", "--", "Status quo: conference"), ("fbs_tiers", "-", "FBS-wide: tier")]:
        if key in POLICIES:
            s_ = MAIN[MAIN.policy == key].groupby("season")[f"group{g + 1}_power"].mean()
            ax.plot(s_.index + 1, s_.values, ls=ls, color=ramp[g], lw=2, label=f"{lab} ranked {g + 1}")
ax.set_xlabel("Season"); ax.set_ylabel("Average power rating of members")
ax.set_title("Strongest four conferences (status quo) vs strongest four FBS-wide tiers", fontsize=11, loc="left")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
plt.savefig(OUT_DIR / "power_comparison_8lines.png", dpi=150, bbox_inches="tight"); plt.show()

# %%
summary = pd.DataFrame({
    "mean_viewers_M": {**{f"{NAMES[k]} (optimized TV)": W[f"optimized:{k}"].mean() for k in POLICIES},
                       **{f"{NAMES[k]} (2019-style TV)": W[f"style2019:{k}"].mean() for k in POLICIES},
                       "2019 televised games by rank, as slotted (A)": W.A_rank.mean(),
                       "Theoretical maximum (F)": W.F_max.mean()}})
summary["index_vs_A"] = 100 * summary.mean_viewers_M / W.A_rank.mean()
summary["index_vs_SQ_optimized"] = 100 * summary.mean_viewers_M / W["optimized:sq"].mean()
tvs = MAIN.groupby("policy")[["never_on_tv", "mean_tv_power", "marquee_games"]].mean()
summary.to_csv(OUT_DIR / "main_summary.csv")
print(summary.round(1).to_string())
print("\nTelevised games (optimized TV), per season:")
print(tvs.rename(index=NAMES).round(1).to_string())
print("\nSettings:", {NAMES[k]: MAIN[MAIN.policy == k].setting.iloc[0] for k in POLICIES})
