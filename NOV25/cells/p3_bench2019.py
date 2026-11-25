# %% [md]
# ## 4. The 2019 benchmark
# Teams: the 130 FBS programs of 2019 in their 2019 conferences. Ratings: 2018 year-end Massey (the preseason
# information for 2019). The actual televised slate is reconstructed from the game file: each game is pinned to its
# real week, day and medium, and the exact MIP recovers its kickoff window (the inventory's seat counts and window power
# totals identify it; the game file itself has no kickoff times).

# %%
TEAMS_2019 = [t for g in ALIGN_2019.values() for t in g]
CONF_2019 = {t: k.split("-")[0] for k, g in ALIGN_2019.items() for t in g}
tid19 = {t: k for k, t in enumerate(TEAMS_2019)}
pw18 = power_df[power_df.Year == 2018].set_index("Team")["Power"]
assert all(t in pw18.index for t in TEAMS_2019)
RHO_2018 = np.array([pw18[t] for t in TEAMS_2019])

fbs19 = games2019[games2019.Team1.isin(tid19) & games2019.Team2.isin(tid19)].copy()
assert fbs19.tv.sum() == len(tv2019), "a televised game involves a non-FBS team"
ALL_PAIRS_2019 = np.sort(np.array([[tid19[a], tid19[b]] for a, b in zip(fbs19.Team1, fbs19.Team2)]), axis=1)
ALL_WEEK_2019 = fbs19.Week.to_numpy()
TV_MASK_2019 = fbs19.tv.to_numpy()
PAIRS_2019 = ALL_PAIRS_2019[TV_MASK_2019]
WEEK_2019 = ALL_WEEK_2019[TV_MASK_2019]
DAY_2019 = fbs19.NumDay.to_numpy()[TV_MASK_2019]
MED_2019 = np.where(fbs19["Shown Air"].to_numpy()[TV_MASK_2019] == 1, 0, 1)        # 0 = broadcast, 1 = cable
print(f"2019: {len(TEAMS_2019)} FBS teams, {len(ALL_PAIRS_2019)} FBS-vs-FBS games, {len(PAIRS_2019)} televised")

# Rating convention check: each Year in the power file is that season's year-end Massey rating, and the 2019 inventory's
# window power totals were built with the 2018 (prior year-end = preseason) ratings of the televised games.
_tot = {y: (tv2019.Team1.map(power_df[power_df.Year == y].set_index("Team").Power)
            + tv2019.Team2.map(power_df[power_df.Year == y].set_index("Team").Power)).sum() for y in (2017, 2018, 2019)}
print("Sum of televised-game power, 2019 slate, by rating year: " + ", ".join(f"{y}: {v:,.2f}" for y, v in _tot.items())
      + f" | inventory: {inventory_df['Total Power'].sum():,.2f}")
assert abs(_tot[2018] - inventory_df["Total Power"].sum()) < 0.01, "inventory not built from 2018 (preseason) ratings"
print(f"Preseason 2019 ratings = 2018 year-end. Top 3: " + ", ".join(f"{t} {v:.2f}" for t, v in pw18.sort_values(ascending=False).head(3).items()))

hist_sched = SeasonScheduler(TEAMS_2019, CONF_2019, scenario_constraints("Actual 2019 slate (reconstructed)"))
HIST = hist_sched.solve(RHO_2018, pairs=PAIRS_2019, pair_week=WEEK_2019, pair_day=DAY_2019, pair_medium=MED_2019)
HIST_ASSIGN = list(zip(HIST.table.pair.astype(int), HIST.table.pos.astype(int)))     # (televised-game index, position)
print(f"Actual 2019 slate: {HIST.stats['games']} games placed, violations={HIST.stats['violations']}, "
      f"caps relaxed={HIST.stats['cap_relaxed_groups']}, realized {HIST.realized:.1f}M total viewers ({HIST.stats['time']:.1f}s)")

# 2019-style rule: appeal rank of every 2019 FBS game; televised games' quantiles and positions, most appealing first
_s_all = SeasonScheduler(TEAMS_2019, CONF_2019, scenario_constraints(None, matchup_mode="subset"))
_s_all._prepare(RHO_2018, ALL_PAIRS_2019, None)
_rank_all = np.empty(len(ALL_PAIRS_2019), int)
_rank_all[np.argsort(-matchup_appeal(_s_all), kind="stable")] = np.arange(len(ALL_PAIRS_2019))
_tv_rank = _rank_all[TV_MASK_2019]                                    # appeal rank of each televised game (TV order)
_pos_tv = np.empty(len(PAIRS_2019), int)
_pos_tv[HIST.table.pair.to_numpy().astype(int)] = HIST.table.pos.to_numpy().astype(int)
SLOT_ORDER_2019 = [((r + 0.5) / len(ALL_PAIRS_2019), int(p_)) for r, p_ in sorted(zip(_tv_rank, _pos_tv))]
_chk = pattern_tv(SeasonScheduler(TEAMS_2019, CONF_2019, scenario_constraints(None, matchup_mode="subset")),
                  RHO_2018, ALL_PAIRS_2019, SLOT_ORDER_2019, pair_week=ALL_WEEK_2019)
print(f"Check: the 2019-style rule applied to the full 2019 schedule reproduces the 2019 slate: "
      f"{_chk.realized:.1f}M vs {HIST.realized:.1f}M")
print(f"2019 selectivity: networks televised {len(PAIRS_2019)} of {len(ALL_PAIRS_2019)} FBS games; of the "
      f"{len(PAIRS_2019)} most appealing games, {(_tv_rank < len(PAIRS_2019)).sum()} were televised; median appeal rank of a "
      f"televised game {np.median(_tv_rank) + 1:.0f}")

# Rank of each 2019 team by preseason (2018 year-end) rating, 0 = best; used for the rank-equivalent slates
RANK_2019 = np.empty(len(TEAMS_2019), int)
RANK_2019[np.argsort(-RHO_2018, kind="stable")] = np.arange(len(TEAMS_2019))
TV_RANK_PAIRS_2019 = RANK_2019[PAIRS_2019]                      # televised games as (rank, rank)
ALL_RANK_PAIRS_2019 = RANK_2019[ALL_PAIRS_2019]                 # every FBS-vs-FBS game as (rank, rank)
HIST_POS = np.empty(len(PAIRS_2019), int)                       # 2019 position of each televised game
HIST_POS[HIST.table.pair.to_numpy().astype(int)] = HIST.table.pos.to_numpy().astype(int)
show = HIST.table.assign(r1=RANK_2019[[tid19[t] for t in HIST.table.team1]] + 1, r2=RANK_2019[[tid19[t] for t in HIST.table.team2]] + 1)
print("\nThe ten most-watched (predicted) 2019 games, with preseason ranks:")
print(show.sort_values("realized_view", ascending=False).head(10)[["week", "day", "time", "medium", "team1", "r1", "team2", "r2", "game_power", "realized_view"]]
      .round(2).to_string(index=False))

# %% [md]
# ### 4.1 Scenario results (2019 teams and ratings)
# "Model 0" moves the 2019 televised games among the days, windows and media of their own week (solved exactly).
# "Model 0b" keeps the whole 2019 FBS slate (723 games, each in its real week) but lets the optimizer choose *which* 303
# to televise and where. Models 2–4 change the matchups themselves. Gains are relative to the actual 2019 slate.

# %%
scen_rows, SCEN_RESULTS = [], {}
for name in CONFIG["scenarios"]:
    cons = scenario_constraints(name)
    sch = SeasonScheduler(TEAMS_2019, CONF_2019, cons)
    mode = cons["matchup_mode"]
    if mode == "subset_week":
        kw = dict(pairs=ALL_PAIRS_2019, pair_week=ALL_WEEK_2019)
    else:
        kw = dict(pairs=PAIRS_2019, pair_week=WEEK_2019, pair_day=DAY_2019, pair_medium=MED_2019)
    if mode == "fixed_week":
        kw["start"] = HIST_ASSIGN
    if mode == "fixed_season":          # start from the best within-week reslotting (any-week moves can only add to it)
        r0 = SCEN_RESULTS["Model 0 (2019 games, reslot within week)"].table
        kw["start"] = list(zip(r0.pair.astype(int), r0.pos.astype(int)))
    res = sch.solve(RHO_2018, seed=SEED, **kw)
    SCEN_RESULTS[name] = res
    scen_rows.append(dict(scenario=name, games=res.stats["games"], linearized_M=res.linear, realized_M=res.realized,
                          gain_vs_2019=100 * (res.realized / HIST.realized - 1), seconds=res.stats["time"],
                          feasible=not res.stats["violations"]))
scen_table = pd.DataFrame(scen_rows)
scen_table.to_csv(OUT_DIR / "scenario_results_2019.csv", index=False)
print(scen_table.round(1).to_string(index=False))

# %% [md]
# ### 4.2 Heuristic quality (MIP benchmark)
# The MIP is restricted to the top-K candidates per position plus the heuristic schedule; its dual bound bounds the best
# schedule *within that pool* (Appendix B.2.3).

# %%
if RUN_MODE != "quick":
    bench_rows = []
    for name in ["Model 4-sigma (no conference structure)", "Model 3-sigma (conference caps)"]:
        sch = SeasonScheduler(TEAMS_2019, CONF_2019, scenario_constraints(name), solver=dict(use_mip=True, top_k=120, mip_time_limit=60))
        r = sch.solve(RHO_2018, seed=SEED)
        st = r.stats
        bench_rows.append(dict(scenario=name, heuristic_M=st["heuristic_linear"], heuristic_s=st["heuristic_time"], final_M=r.linear,
                               pool_vars=st.get("mip_vars"), pool_bound_M=st.get("mip_bound"), mip_s=st.get("mip_time"),
                               heuristic_gap_pct=100 * (st.get("mip_bound", np.nan) / st["heuristic_linear"] - 1)))
    print(pd.DataFrame(bench_rows).round(2).to_string(index=False))
