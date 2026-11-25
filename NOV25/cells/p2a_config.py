# %% [md]
# ## 3. The season scheduler
#
# ### 3.1 Constraint configuration
# All constraints, conference structures, time-zone lists, solver and simulation settings live in
# **`schedule_config.json`** (written on first run; edit the file, not this cell). A *scenario* is a named set of
# overrides on top of `"defaults"`.
#
# | Key | Paper eq. | Meaning |
# |---|---|---|
# | `matchup_mode` | (A.10)–(A.11) | `free` (any pair), `subset` (choose among a given list, any week), `subset_week` (choose among a given list, each game in its own week), `fixed_week` (each listed game exactly once, in its week), `fixed_season` (each listed game once, any week), `historical` (each game on its own week, day and medium) |
# | `max_games_per_team` | (A.5) | Upper bound on televised games per team |
# | `one_game_per_week` | (A.3) | A team appears in at most one televised game per week |
# | `rest_rule` | (A.6) | No late-week game followed by an early-week game the next week |
# | `fill_all_seats` | (A.4) | Fill every seat of the broadcast inventory (303 in 2019) |
# | `power_cap_slack` | (A.8) | σ: total power in a window ≤ σ × the window's reference power |
# | `min_exposure` | (A.13) | Every team appears in at least this many televised games |
# | `team_group_caps` | (A.12) | Per-team caps on televised games against a group |
# | `time_zone_caps` | (A.14) | Caps on games involving a time zone in a kickoff window |
# | `brand_mode` | — | `team` (Notre Dame/BYU own brand, others conference), `conference`, `none`, or `average` |
#
# **Window reference power.** Constraint (A.8) and the linearization of the objective need a reference total power
# for each window. For the 2019 benchmark it is the 2019 window total, as before. In the simulation it is, by default,
# the total power *this season* of the 2019 televised games at the same preseason ranks in that window
# (`cap_reference = "rank_2019"`): "if Saturday prime time carried games of ranks 3, 9, 14, 40 and 77 in 2019, it may
# carry at most as much combined power as this season's teams at those ranks." This scales the cap with the season's
# rating distribution and makes the rank-equivalent 2019 slate feasible by construction.

# %%
ALIGN_2019 = {
    "American_Athletic-east": ["CIN", "UCF", "ECU", "Temple", "USF", "Connecticut"],
    "American_Athletic-west": ["TULANE", "Houston", "SMU", "Navy", "Memphis", "Tulsa"],
    "Atlantic_Coast-atlantic": ["CLEM", "Florida State", "Louisville", "N.C. State", "Syracuse", "Wake Forest", "BC"],
    "Atlantic_Coast-coastal": ["North Carolina", "Duke", "Pittsburgh", "Georgia Tech", "Miami", "Virginia", "Virginia Tech"],
    "Big_10-east": ["Indiana", "Maryland", "Michigan", "Michigan State", "Ohio State", "Penn State", "Rutgers"],
    "Big_10-west": ["Illinois", "Iowa", "Minnesota", "Nebraska", "Northwestern", "Purdue", "Wisconsin"],
    "Big_12": ["BAY", "Iowa State", "KANSAS", "Kansas State", "Oklahoma", "Oklahoma State", "TCU", "Texas", "Texas Tech", "West Virginia"],
    "Conference_USA-east": ["Charlotte", "Florida Atlantic", "FIU", "Middle Tennessee", "Western Kentucky", "MARSHALL", "Old Dominion"],
    "Conference_USA-west": ["Louisiana Tech", "North Texas", "Rice", "UAB", "UTSA", "UTEP", "Southern Miss"],
    "FBS_Indep": ["Army", "Liberty", "UMASS", "N. Mexico St."],
    "Mid_American-east": ["Akron", "BGSU", "BUF", "Kent State", "Miami (Ohio)", "Ohio"],
    "Mid_American-west": ["BALL", "C. Mich.", "E. Michigan", "Northern Illinois", "Toledo", "Western Michigan"],
    "Mountain_West-mountain": ["AF", "Boise", "Colorado St.", "NM", "Utah State", "Wyoming"],
    "Mountain_West-west": ["Fresno State", "Hawaii", "Nevada", "San Diego State", "San Jose State", "UNLV"],
    "Pac_12-north": ["Cal", "Oregon", "Oregon State", "Stanford", "Washington", "Washington State"],
    "Pac_12-south": ["ARIZ", "Arizona St.", "Colorado", "UCLA", "USC", "Utah"],
    "Southeastern-east": ["FLA", "Georgia", "Kentucky", "Missouri", "South Carolina", "Tennessee", "Vanderbilt"],
    "Southeastern-west": ["ALA", "ARK", "AUB", "LSU", "Mississippi", "Mississippi State", "Texas A&M"],
    "Sun_Belt-east": ["APP ST", "TROY", "C CAR", "Georgia Southern", "Georgia St."],
    "Sun_Belt-west": ["ARK ST", "UL-Lafayette", "South Alabama", "Texas State", "UL-Monroe"],
    "BYU": ["BYU"],
    "NotreDame": ["Notre Dame"],
}

ALIGN_2025 = {   # as in the Hottovy simulator (134 teams)
    "Southeastern": ["ALA", "ARK", "AUB", "FLA", "Georgia", "Kentucky", "LSU", "Mississippi", "Mississippi State", "Missouri",
                     "Oklahoma", "South Carolina", "Tennessee", "Texas", "Texas A&M", "Vanderbilt"],
    "Big_10": ["Illinois", "Indiana", "Iowa", "Maryland", "Michigan", "Michigan State", "Minnesota", "Nebraska", "Northwestern",
               "Ohio State", "Oregon", "Penn State", "Purdue", "Rutgers", "UCLA", "USC", "Washington", "Wisconsin"],
    "Big_12": ["ARIZ", "Arizona St.", "BAY", "BYU", "CIN", "Colorado", "Houston", "Iowa State", "KANSAS", "Kansas State",
               "Oklahoma State", "TCU", "Texas Tech", "UCF", "Utah", "West Virginia"],
    "Atlantic_Coast": ["BC", "Cal", "CLEM", "Duke", "Florida State", "Georgia Tech", "Louisville", "Miami", "N.C. State",
                       "North Carolina", "Pittsburgh", "SMU", "Stanford", "Syracuse", "Virginia", "Virginia Tech", "Wake Forest"],
    "Pac_12": ["Oregon State", "Washington State"],
    "Mountain_West": ["AF", "Boise", "Colorado St.", "Fresno State", "Hawaii", "Nevada", "NM", "San Diego State",
                      "San Jose State", "UNLV", "Utah State", "Wyoming"],
    "American_Athletic": ["Army", "Charlotte", "ECU", "Florida Atlantic", "Memphis", "Navy", "North Texas", "Rice", "Temple",
                          "TULANE", "Tulsa", "UAB", "USF", "UTSA"],
    "Sun_Belt": ["APP ST", "ARK ST", "C CAR", "Georgia Southern", "Georgia St.", "J. Madison", "MARSHALL", "Old Dominion",
                 "South Alabama", "Southern Miss", "Texas State", "TROY", "UL-Lafayette", "UL-Monroe"],
    "Mid_American": ["Akron", "BALL", "BGSU", "BUF", "C. Mich.", "E. Michigan", "Kent State", "Miami (Ohio)",
                     "Northern Illinois", "Ohio", "Toledo", "Western Michigan"],
    "Conference_USA": ["FIU", "Jacksonville St.", "Kennesaw St.", "Liberty", "Louisiana Tech", "Middle Tennessee",
                       "N. Mexico St.", "Sam Houston St.", "UTEP", "Western Kentucky"],
    "FBS_Indep": ["Connecticut", "UMASS", "Notre Dame"],
}

# Per-team caps on televised games against a group (from NcaaJan20 `confrest` / `outsideconfrest`, with the two fixes noted above).
DIV_CAPS = [
    ("American_Athletic-east", "American_Athletic-east", 5), ("American_Athletic-west", "American_Athletic-west", 5),
    ("American_Athletic-east", "American_Athletic-west", 3), ("American_Athletic-west", "American_Athletic-east", 3),
    ("Atlantic_Coast-atlantic", "Atlantic_Coast-atlantic", 6), ("Atlantic_Coast-coastal", "Atlantic_Coast-coastal", 6),
    ("Atlantic_Coast-atlantic", "Atlantic_Coast-coastal", 2), ("Atlantic_Coast-coastal", "Atlantic_Coast-atlantic", 2),
    ("Big_10-east", "Big_10-east", 6), ("Big_10-west", "Big_10-west", 6), ("Big_10-west", "Big_10-east", 3), ("Big_10-east", "Big_10-west", 3),
    ("Big_12", "Big_12", 9),
    ("Conference_USA-east", "Conference_USA-east", 6), ("Conference_USA-west", "Conference_USA-west", 6),
    ("Conference_USA-west", "Conference_USA-east", 2), ("Conference_USA-east", "Conference_USA-west", 2),
    ("Mid_American-east", "Mid_American-east", 5), ("Mid_American-west", "Mid_American-west", 5),
    ("Mid_American-west", "Mid_American-east", 3), ("Mid_American-east", "Mid_American-west", 3),
    ("Mountain_West-mountain", "Mountain_West-mountain", 5), ("Mountain_West-west", "Mountain_West-west", 5),
    ("Mountain_West-mountain", "Mountain_West-west", 3), ("Mountain_West-west", "Mountain_West-mountain", 3),
    ("Pac_12-north", "Pac_12-north", 5), ("Pac_12-south", "Pac_12-south", 5), ("Pac_12-south", "Pac_12-north", 4), ("Pac_12-north", "Pac_12-south", 4),
    ("Southeastern-east", "Southeastern-east", 6), ("Southeastern-west", "Southeastern-west", 6),
    ("Southeastern-west", "Southeastern-east", 2), ("Southeastern-east", "Southeastern-west", 2),
    ("Sun_Belt-east", "Sun_Belt-east", 4), ("Sun_Belt-west", "Sun_Belt-west", 4), ("Sun_Belt-west", "Sun_Belt-east", 4), ("Sun_Belt-east", "Sun_Belt-west", 4),
]
OUTSIDE_CAPS = {"American_Athletic": 4, "Atlantic_Coast": 4, "Big_10": 3, "Big_12": 3, "Conference_USA": 4, "FBS_Indep": 12,
                "Mid_American": 4, "Mountain_West": 4, "Pac_12": 3, "Southeastern": 4, "Sun_Belt": 4, "BYU": 12, "NotreDame": 12}


def _groups_2019():
    g = dict(ALIGN_2019)
    for conf in {k.split("-")[0] for k in ALIGN_2019}:
        g.setdefault(conf, sorted({t for k, v in ALIGN_2019.items() if k.split("-")[0] == conf for t in v}))
    return g


DEFAULT_CONFIG = {
    "version": 3,
    "calendar": {"early_days": [1, 2, 3, 4, 5], "late_days": [6, 7], "last_week": 14,
                 "day_codes": "1=Sun 2=Mon 3=Tue 4=Wed 5=Thu 6=Fri 7=Sat"},
    "defaults": {
        "matchup_mode": "free", "max_games_per_team": 12, "one_game_per_week": True, "rest_rule": True,
        "fill_all_seats": True, "power_cap_slack": 1.0, "low_power_window": 50, "min_exposure": 0,
        "team_group_caps": None, "time_zone_caps": None, "brand_mode": "team",
    },
    "solver": {"grasp_starts": 1, "rcl_noise": 0.03, "improve_passes": 4, "swap_moves": True,
               "fill_margin": 6.0, "use_mip": False, "mip_backend": "auto", "mip_time_limit": 30, "mip_gap": 0.015, "top_k": 120,
               "exact_fixed": True},
    # Simulation-level settings (Sections 7-10). cap_reference: "rank_2019" = each window's power cap is the power of the
    # 2019 televised games at the same preseason ranks this season; "absolute_2019" = the 2019 window power as recorded.
    "simulation": {"cap_reference": "rank_2019", "power_cap_slack": 1.0, "season_games": 12, "sigma_game": 15.5,
                   "conference_games": {"Southeastern": 9, "Big_10": 9, "Big_12": 9, "Atlantic_Coast": 8, "American_Athletic": 8,
                                        "Sun_Belt": 8, "Mid_American": 8, "Mountain_West": 8, "Conference_USA": 8,
                                        "Pac_12": 1, "FBS_Indep": 0}},
    "alignments": {"2019": ALIGN_2019, "2025": ALIGN_2025},
    "groups": {"2019": _groups_2019()},
    "team_group_cap_sets": {
        "conference_2019": {
            "groups": "2019",
            "rules": [{"team_group": a, "opp_group": b, "max": k} for a, b, k in DIV_CAPS]
                     + [{"team_group": c, "opp_not_group": c, "max": k} for c, k in OUTSIDE_CAPS.items()],
        }
    },
    "time_zones": {
        "West": ["AF", "ARIZ", "Arizona St.", "Boise", "BYU", "Cal", "Colorado", "Colorado St.", "Fresno State", "Hawaii", "Idaho",
                 "N. Mexico St.", "Nevada", "NM", "Oregon", "Oregon State", "San Diego State", "San Jose State", "Stanford", "UCLA",
                 "UNLV", "USC", "Utah", "Utah State", "UTEP", "Washington", "Washington State", "Wyoming"],
        "Central": ["ALA", "ARK ST", "AUB", "BALL", "BAY", "CIN", "Houston", "Illinois", "Iowa", "Iowa State", "KANSAS", "Kansas State",
                    "LSU", "Memphis", "Michigan", "Middle Tennessee", "Minnesota", "Mississippi", "Mississippi State", "Missouri", "Nebraska",
                    "North Texas", "Northern Illinois", "Northwestern", "Notre Dame", "Oklahoma", "Oklahoma State", "Rice", "SMU",
                    "South Alabama", "Southern Miss", "TCU", "Texas", "Texas A&M", "Texas State", "Texas Tech", "TROY", "TULANE", "Tulsa",
                    "UAB", "UL-Lafayette", "UL-Monroe", "UTSA", "Vanderbilt", "Western Kentucky", "Wisconsin", "Jacksonville St.",
                    "Sam Houston St."],
        "_default": "East",
    },
    "time_zone_cap_sets": {"mild_2019": [{"zone": "West", "time": "morning", "max": 2},
                                         {"zone": "Central", "time": "late", "max": 6},
                                         {"zone": "East", "time": "late", "max": 2}]},
    "scenarios": {
        "Actual 2019 slate (reconstructed)": {"matchup_mode": "historical", "rest_rule": False},
        "Model 0 (2019 games, reslot within week)": {"matchup_mode": "fixed_week", "rest_rule": False},
        "Model 0b (2019 slate, choose TV games within week)": {"matchup_mode": "subset_week", "rest_rule": False},
        "Model 1 (2019 games by week, sigma=1.05)": {"matchup_mode": "fixed_week", "power_cap_slack": 1.05, "rest_rule": False},
        "Model 2 (2019 games, any week)": {"matchup_mode": "fixed_season", "rest_rule": False},
        "Model 3-sigma (conference caps)": {"team_group_caps": "conference_2019"},
        "Model 3-zeta-sigma": {"team_group_caps": "conference_2019", "time_zone_caps": "mild_2019"},
        "Model 3-eps-sigma": {"team_group_caps": "conference_2019", "min_exposure": 1},
        "Model 3-zeta-eps-sigma": {"team_group_caps": "conference_2019", "time_zone_caps": "mild_2019", "min_exposure": 1},
        "Model 4-sigma (no conference structure)": {},
        "Model 4-sigma-zeta": {"time_zone_caps": "mild_2019"},
        "Model 4-sigma-eps": {"min_exposure": 1},
        "Model 4-sigma-zeta-eps": {"time_zone_caps": "mild_2019", "min_exposure": 1},
    },
}

cfg_path = find_file("config", required=False)
if cfg_path is not None and json.loads(Path(cfg_path).read_text()).get("version", 0) < DEFAULT_CONFIG["version"]:
    Path(cfg_path).rename(Path(cfg_path).with_suffix(".old.json"))   # keep the user's edits, regenerate the new format
    print(f"Older config format found; saved as {Path(cfg_path).with_suffix('.old.json')}")
    cfg_path = None
if cfg_path is None:
    cfg_path = DATA_DIR / "schedule_config.json"
    cfg_path.write_text(json.dumps(DEFAULT_CONFIG, indent=1))
    print(f"Wrote default configuration to {cfg_path}")
CONFIG = json.loads(Path(cfg_path).read_text())
print(f"Loaded {cfg_path}: {len(CONFIG['scenarios'])} scenarios, cap sets {list(CONFIG['team_group_cap_sets'])}, "
      f"time-zone cap sets {list(CONFIG['time_zone_cap_sets'])}")


def scenario_constraints(name=None, **overrides):
    """Merge defaults, a named scenario and keyword overrides into one constraint dict."""
    c = dict(CONFIG["defaults"])
    if name:
        c.update(CONFIG["scenarios"][name])
    c.update(overrides)
    return c
