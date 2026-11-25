# %% [md]
# # Optimizing NCAA FBS Viewership: integrated pipeline (version 3b: no conference brands)
#
# **Version 3b** reruns everything with the audience model re-estimated **without conference terms** (the
# "W/O Conferences" column of `with_and_without_conferences.xlsx`). No team carries a conference, Notre Dame or BYU
# brand effect; conference membership affects only who plays whom.
#
# This notebook runs the whole study top to bottom: viewership model → season scheduler → team-strength
# dynamics → league policies → 50-season simulation → decomposition of the gains → policy rulebook.
#
# **What is new in version 3 (relative to the prototype notebook).**
#
# 1. **A realistic status quo.** The prototype compared policies with a status quo whose television schedule was
#    already *optimized* (the best 303 of ~800 games, optimally slotted). That made the status quo look 35–55% better
#    than the 2019 slate and left policies only ~5% of room. Version 3 adds, every simulated season, the slate you
#    would get if networks televised **the same games as in 2019 by preseason rank** (2019's #1 vs #32 becomes this
#    season's #1 vs #32), in the same week, day, window and medium.
# 2. **A decomposition of the gains** into *optimization* (which games are televised and where) and *policy* (which
#    games exist), as a 2 × 2 design: {status quo, policy} × {2019-style TV choices, optimized TV}.
# 3. **Policies written as rules a league office could run**: fixed tier sizes, a fixed number of teams promoted and
#    relegated by standings, and simple pairing rules. The tuned settings are printed as a plain-language rulebook
#    (Section 11) and saved to `outputs/policy_rulebook.md`.
# 4. **Fixes**: every team now plays exactly 12 games under every policy (the prototype left some teams with 6);
#    fixed-matchup reslotting is solved exactly (one small MIP per week); the window power cap is defined per season
#    from the 2019 slate at the same ranks, so it scales with the season's rating distribution.
#
# | Section | Content |
# |---|---|
# | 1 | Setup and run size |
# | 2 | Data and the log-linear viewership model (checked against the paper) |
# | 3 | Season scheduler: constraints file, optimizer, exact reslotting, 2019-style TV rule |
# | 4 | 2019 benchmark: actual slate, reslotting and structural scenarios |
# | 5 | Team-strength dynamics and validation |
# | 6 | League policies (implementable rules) |
# | 7 | Season simulator and the decomposition ladder |
# | 8 | Experimental design: tuning each policy |
# | 9 | Main experiment: where the gains come from |
# | 10 | Robustness: power cap σ, minimum exposure, lagged ratings |
# | 11 | Policy rulebook (generated from the tuned settings) |
# | 12 | Summary of findings and notes for the paper |
#
# **Inputs** (in `DATA_DIR`): `model coefficients.csv`, `wk_day_slot_summary_NO_FCS.csv`, `rivalry for IP.csv`,
# `PwrRankings for IP.csv`, `all2019games - NO FCS.csv`, `schedule_config.json` (written on first run if absent).
# Optional: `massey_1960_2019.csv` (columns `Team, Year, Power`) for the paper's 60-season validation.
#
# **Run size.** `RUN_MODE = "quick"` (smoke test, ~3 min), `"standard"` (~25 min) or `"paper"` (~1.5 h).

# %%
import json, time, math, itertools, warnings, zlib
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse, stats
from scipy.optimize import milp, LinearConstraint, Bounds

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

DATA_DIR = Path("data")
# Audience-model switch. Default: no conference brand terms (version 3b). Set WITH_CONFERENCES = True (or run
# `python run_all.py standard --with-conferences`) to use the published coefficients with conference, Notre Dame and
# BYU terms; results then go to outputs_with_conferences/ and the brand-neutral robustness check is added.
import os
WITH_CONFERENCES = bool(globals().get("WITH_CONFERENCES", os.environ.get("FBS_WITH_CONFERENCES", "0") == "1"))
OUT_DIR = Path("outputs_with_conferences" if WITH_CONFERENCES else "outputs")
OUT_DIR.mkdir(exist_ok=True)

RUN_MODE = globals().get("RUN_MODE_OVERRIDE", "standard")      # "quick" | "standard" | "paper"
SEED = 2026

RUN = {
    #            seasons reps  DOE seasons DOE reps  robustness reps  GRASP starts
    "quick":    dict(seasons=12, reps=2,  doe_seasons=8,  doe_reps=1, rob_reps=1, grasp_starts=1),
    "standard": dict(seasons=50, reps=6,  doe_seasons=25, doe_reps=2, rob_reps=2, grasp_starts=1),
    "paper":    dict(seasons=50, reps=20, doe_seasons=30, doe_reps=4, rob_reps=5, grasp_starts=2),
}[RUN_MODE]
print(f"RUN_MODE={RUN_MODE}: {RUN}")

try:
    import gurobipy as gp  # noqa: F401
    HAVE_GUROBI = True
except Exception:
    HAVE_GUROBI = False
print("Gurobi available:", HAVE_GUROBI, "| MIP solver otherwise: SciPy/HiGHS")

FILE_ALIASES = {
    "coefficients": (["model coefficients (with conferences).csv"] if WITH_CONFERENCES
                     else ["model coefficients.csv", "model_coefficients.csv"]),
    "inventory":    ["wk_day_slot_summary_NO_FCS.csv", "wk_day_slot_summary_-_NO_FCS.csv"],
    "rivalries":    ["rivalry for IP.csv", "rivalry_for_IP.csv"],
    "power":        ["PwrRankings for IP.csv", "PwrRankings_for_IP.csv"],
    "games2019":    ["all2019games - NO FCS.csv", "all2019games_-_NO_FCS.csv"],
    "massey_hist":  ["massey_1960_2019.csv"],
    "config":       ["schedule_config.json"],
}


def find_file(key, required=True):
    for name in FILE_ALIASES[key]:
        for folder in (DATA_DIR, Path(".")):
            if (folder / name).exists():
                return folder / name
    if required:
        raise FileNotFoundError(f"Missing input '{key}'. Expected one of {FILE_ALIASES[key]} in {DATA_DIR.resolve()}")
    return None


for k in FILE_ALIASES:
    p = find_file(k, required=False)
    print(f"  {k:<13} {'found: ' + str(p) if p else 'not found' + (' (optional)' if k in ('massey_hist', 'config') else ' (REQUIRED)')}")

# %% [md]
# ## 2. Data and the log-linear viewership model
#
# $\ln \hat v = \beta_0 + \beta_{\text{year}} + \beta_{\text{day}} + \beta_{\text{time}} + \beta_{\text{week}} + \beta_{\text{medium}}
#  + b_i + b_j + \beta_G G + \beta_P P + \beta_F F + \beta_\Delta \Delta + \beta_{FG} F G + \beta_{FP} F P + \beta_R R$,
# with $P=\rho_i+\rho_j$ (game power), $\Delta=|\rho_i-\rho_j|$, $G$ = games in the window,
# $F=\min\{100P/(P^{\text{tot}}+0.01),100\}$ (the game's share of the window's total power), $R$ = rivalry and
# $b_i$ = brand, which is **zero for every team in this version** (no conference, Notre Dame or BYU terms). Viewership is in millions; the season effect is held at
# 2018 (the last estimation season), as in the paper.
#
# Calendar coding (verified against the 2019 game file): days 1 = Sunday … 6 = Friday, 7 = Saturday (baseline);
# week 1 is the baseline week, weeks 0 and 2–5 have their own effects, 6–9 are "mid-season", 10–13 are "rivals-4" …
# "rivals-1" and 14 is "rivals".
#
# The 2019 game file lists 829 games, of which 106 involve an FCS opponent (none televised); they are dropped.

# %%
coef_df = pd.read_csv(find_file("coefficients"))
coef_df = coef_df.rename(columns={coef_df.columns[0]: "term"}).dropna(subset=["term"])
COEF = dict(zip(coef_df["term"].str.strip(), coef_df["Estimate"].astype(float)))

inventory_df = pd.read_csv(find_file("inventory")).loc[:, ["Week", "NumDay", "Slot", "Num Games", "Num Broadcast", "Num Cable", "Total Power"]]
inventory_df["Slot"] = inventory_df["Slot"].str.strip()
assert (inventory_df["Num Games"] == inventory_df["Num Broadcast"] + inventory_df["Num Cable"]).all()

power_df = pd.read_csv(find_file("power")).loc[:, ["Team", "Power", "Year"]]
power_df["Team"] = power_df["Team"].str.strip()

rival_df = pd.read_csv(find_file("rivalries"))
RIVALRIES = {frozenset((a.strip(), b.strip())) for a, b in zip(rival_df.Team1, rival_df.Team2)}

games2019 = pd.read_csv(find_file("games2019"))
games2019 = games2019.rename(columns={games2019.columns[4]: "Day"})
for c in ("Team1", "Team2"):
    games2019[c] = games2019[c].str.strip()
games2019["Shown Air"] = games2019["Shown Air"].fillna(0).astype(int)
games2019["Shown Cable"] = games2019["Shown Cable"].fillna(0).astype(int)
dow = (games2019["Date"].to_numpy() - 43708) % 7                    # Excel serial 43708 = Saturday 31 Aug 2019
games2019["NumDay"] = [{0: 7, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}[d] for d in dow]
games2019["tv"] = (games2019["Shown Air"] == 1) | (games2019["Shown Cable"] == 1)
tv2019 = games2019[games2019.tv].copy()

print(f"Coefficients: {len(COEF)} | windows: {len(inventory_df)} | televised seats: {inventory_df['Num Games'].sum()} "
      f"(broadcast {inventory_df['Num Broadcast'].sum()}, cable {inventory_df['Num Cable'].sum()})")
print(f"Power ratings: {power_df.Year.min()}–{power_df.Year.max()}, {power_df.Team.nunique()} teams | rivalries: {len(RIVALRIES)} | "
      f"2019 games in file: {len(games2019)}, televised: {len(tv2019)}")

# %%
TIMES = ["morning", "afternoon", "evening", "late"]
MEDIA = ["broadcast", "cable"]
CONFERENCES = ["American_Athletic", "Atlantic_Coast", "Big_10", "Big_12", "Conference_USA", "FBS_Indep", "Mid_American",
               "Mountain_West", "Pac_12", "Southeastern", "Sun_Belt"]


class ViewershipModel:
    """Log-linear viewership model of Park, Kenter & Hasler (2025), coefficients from `model coefficients.csv`."""

    def __init__(self, coef, year_term="Year2018", last_week=14):
        c = self.coef = coef
        self.intercept = c["(Intercept)"] + (c.get(year_term, 0.0) if year_term else 0.0)
        self.day = {d: c.get(f"NumDay{d}", 0.0) for d in range(1, 8)}
        self.time = {"morning": 0.0, "afternoon": c["timeslotafternoon"], "evening": c["timeslotevening"], "late": c["timeslotlate"]}
        self.medium = {"broadcast": c["broadcast"], "cable": 0.0}
        self.last_week = last_week
        self.b_games, self.b_power = c["NumGames"], c["currpwr"]
        self.b_frac, self.b_diff = c["FractionPwr"], c["pwrdiff"]
        self.b_frac_games, self.b_frac_power = c["FracPwr_int_NumGames"], c["FracPwr_int_currpwr"]
        self.b_rivalry = c["rivalry"]
        self.conference = {k: c[k] for k in CONFERENCES}
        self.team_override = {"Notre Dame": c["NotreDame"], "BYU": c["BYU"]}

    def week(self, w):
        L = self.last_week
        if w == 1:
            return 0.0
        if w == 0 or 2 <= w <= 5:
            return self.coef.get(f"WEEK{w}", 0.0)
        if w == L:
            return self.coef["WEEKrivals"]
        if L - 4 <= w <= L - 1:
            return self.coef[f"WEEKrivals-{L - w}"]
        return self.coef["WEEKmid_season"]

    def window_const(self, w, d, t):
        return self.intercept + self.week(w) + self.day[d] + self.time[t]

    def brand(self, team, conf, mode="team"):
        """'team': Notre Dame and BYU keep their own effect, others use their conference; 'conference'; 'none'."""
        if mode == "none":
            return 0.0
        if mode == "team" and team in self.team_override:
            return self.team_override[team]
        return self.conference.get(conf, 0.0)

    def log_view(self, P, D, brand_sum, riv, G, Ptot, wconst, mconst):
        F = np.minimum(100.0 * P / (Ptot + 0.01), 100.0)
        return (wconst + mconst + brand_sum + self.b_rivalry * riv + self.b_games * G + self.b_power * P
                + self.b_frac * F + self.b_diff * D + self.b_frac_games * F * G + self.b_frac_power * F * P)


VM = ViewershipModel(COEF)
BRAND_TERMS = CONFERENCES + ["NotreDame", "BYU"]
if WITH_CONFERENCES:
    assert any(COEF[k] != 0 for k in BRAND_TERMS), "WITH_CONFERENCES needs the coefficient file with conference terms"
    print("WITH-CONFERENCE audience model:", sum(COEF[k] != 0 for k in BRAND_TERMS), "non-zero brand terms; outputs ->", OUT_DIR)
else:
    assert all(COEF[k] == 0 for k in BRAND_TERMS), "this version expects the no-conference coefficient file"
    print("No-conference audience model: all", len(BRAND_TERMS), "brand terms are zero")

check = pd.DataFrame({
    "effect": ["Friday", "Thursday", "Monday", "Afternoon", "Evening", "Late", "Over-the-air", "Week 7 (mid)", "Week 14 (rivals)",
               "Notre Dame brand", "SEC brand", "Big Ten brand"],
    "log effect": [VM.day[6], VM.day[5], VM.day[2], VM.time["afternoon"], VM.time["evening"], VM.time["late"], VM.medium["broadcast"],
                   VM.week(7), VM.week(14), VM.brand("Notre Dame", "FBS_Indep"), VM.conference["Southeastern"], VM.conference["Big_10"]]})
check["% change"] = 100 * (np.exp(check["log effect"]) - 1)
print(check.round(3).to_string(index=False))


def pct(a, b):
    return 100 * (np.exp(a - b) - 1)


base = dict(brand_sum=0.0, riv=0, wconst=0.0, mconst=0.0, D=0.0)
g_lo = VM.log_view(P=110, G=5, Ptot=110 / 0.20 - 0.01, **base)
g_hi = VM.log_view(P=120, G=5, Ptot=120 / 0.20 - 0.01, **base)
v = {g: VM.log_view(P=110, G=g, Ptot=110 * g, **base) for g in (2, 3, 4, 5)}
paper_checks = pd.DataFrame({
    "quantity": ["+10 Game Power (P=110, F held at 20)", "Over-the-air vs cable", "Evening vs morning", "Rivalry",
                 "Add 3rd typical game (per game)", "Add 5th typical game (per game)", "+14 Power Diff"],
    "with-conference model": ["+22.7% to +24.7%", "+134.7%", "+16.1%", "+10.7%", "−15.9%", "−9.8%", "+11.6%"],
    "notebook": [pct(g_hi, g_lo), pct(VM.medium["broadcast"], 0), pct(VM.time["evening"], 0), pct(VM.b_rivalry, 0),
                 pct(v[3], v[2]), pct(v[5], v[4]), pct(14 * VM.b_diff, 0)]})
paper_checks["notebook"] = paper_checks["notebook"].map(lambda x: f"{x:+.1f}%")
print()
print(paper_checks.to_string(index=False))
