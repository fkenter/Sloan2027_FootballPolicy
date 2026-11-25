# %% [md]
# ## 5. Team-strength dynamics (smoothed regime-switching AR(3))
#
# For each team, rating = league center + $Z_t$, with $Z_t=\tilde\mu_t+X_t$:
# * **Era baseline.** Every 5–10 seasons (uniform) a new era level $c_k$ is drawn; the baseline follows it with exponential
#   smoothing, $\tilde\mu_t=\alpha c_k+(1-\alpha)\tilde\mu_{t-1}$, $\alpha=0.4$.
# * **Momentum.** $X_t=0.55X_{t-1}+0.20X_{t-2}+0.10X_{t-3}+\varepsilon_t$, $\varepsilon_t\sim N(0,4.5^2)$.
# * Each season the ratings are re-centered on the league center and clipped to **[35, 85]**.
#
# **What changed relative to the Hottovy file, and why.**
# 1. *Teams keep their identity.* In the Hottovy code each team's baseline and AR memory started at random values, so a team's
#    actual rating was discarded after season 1 (by the final season Army and Coastal Carolina ranked above Alabama). Here the
#    simulation starts from each team's real rating: the current era level equals its current deviation, and the remaining
#    length of the current era is drawn uniformly from 1–10 seasons.
# 2. *Eras are drawn around each program's long-run level.* With era levels drawn from $N(0,5^2)$ for every team, the model's
#    long-run spread across teams is SD ≈ 8.6, versus 11.5 in the data, because real programs differ permanently (resources,
#    brand). New era levels are therefore drawn as $c_k\sim N(\lambda m_i,\sigma_\mu^2)$, where $m_i$ is the program's average
#    centered rating (2011–2019) and $\lambda$ is calibrated so the long-run cross-sectional SD matches the data. This leaves
#    the *within-team* variance and autocorrelation that Scott validated unchanged (a constant shift does not affect either).
#    Setting `anchor_weight = 0` recovers the description in the paper exactly.
# 3. *League center* defaults to the mean rating of the initial season (≈52), not 55, so the viewership model is evaluated
#    on the scale it was estimated on. The [35, 85] bounds are kept.
#
# **Paper wording (Section 5, item 1):** "…the baseline moves to a new, randomly drawn level *centered on the
# program's long-run average*…"

# %%
DYN = dict(phi=(0.55, 0.20, 0.10), sigma_eps=4.5, sigma_mu=5.0, alpha=0.4, era_min=5, era_max=10,
           p_min=35.0, p_max=85.0, center=None, anchor_weight="calibrate")

# Initial ratings for the 2025 alignment: 2019 year-end Massey (last full pre-pandemic season); teams new to FBS get their
# conference's median.
TEAMS_SIM = [t for g in ALIGN_2025.values() for t in g]
CONF_SIM = {t: c for c, g in ALIGN_2025.items() for t in g}
pw19 = power_df[power_df.Year == 2019].set_index("Team")["Power"]
init = pd.Series({t: pw19.get(t, np.nan) for t in TEAMS_SIM})
conf_med = init.groupby(pd.Series(CONF_SIM)).median()
no_rating = init[init.isna()].index.tolist()
init = init.fillna(pd.Series(CONF_SIM).map(conf_med))
RHO_INIT = init.to_numpy()
print(f"{len(TEAMS_SIM)} teams; given their conference median (no 2019 rating): {no_rating}")

# Long-run program level m_i: mean of the season-centered rating, 2011–2019
hist = power_df[(power_df.Year >= 2011) & (power_df.Year <= 2019)].copy()
hist["Z"] = hist.Power - hist.groupby("Year").Power.transform("mean")
m_hist = hist.groupby("Team").Z.mean()
LONG_RUN = np.array([m_hist.get(t, RHO_INIT[k] - RHO_INIT.mean()) for k, t in enumerate(TEAMS_SIM)])
TARGET_SD = hist.groupby("Year").Z.std().mean()


def simulate_power(init_rating, long_run, n_seasons, n_reps, seed=0, p=DYN, anchor=None):
    """Vectorized team-strength simulator.

    Returns an array of shape (n_reps, n_seasons, n_teams). Season 0 is the (clipped) initial rating.
    All teams and replications are advanced together, so 50 seasons × 20 replications × 134 teams takes milliseconds.
    """
    rng = np.random.default_rng(seed)
    init_rating = np.asarray(init_rating, float)
    N, R = len(init_rating), n_reps
    lam = p["anchor_weight"] if anchor is None else anchor
    center = init_rating.mean() if p["center"] is None else p["center"]
    phi1, phi2, phi3 = p["phi"]
    z0 = init_rating - init_rating.mean()
    mu = np.repeat(z0[None, :], R, 0)                     # smoothed era baseline starts at the current level
    target = mu.copy()
    years_left = rng.integers(1, p["era_max"] + 1, (R, N))
    x1 = np.zeros((R, N)); x2 = np.zeros((R, N)); x3 = np.zeros((R, N))
    out = np.empty((R, n_seasons, N))
    out[:, 0, :] = np.clip(center + z0, p["p_min"], p["p_max"])
    for s in range(1, n_seasons):
        years_left -= 1
        new = years_left <= 0
        if new.any():
            cols = np.nonzero(new)[1]
            target[new] = lam * long_run[cols] + rng.normal(0, p["sigma_mu"], new.sum())
            years_left[new] = rng.integers(p["era_min"], p["era_max"] + 1, new.sum())
        mu = p["alpha"] * target + (1 - p["alpha"]) * mu
        x = phi1 * x1 + phi2 * x2 + phi3 * x3 + rng.normal(0, p["sigma_eps"], (R, N))
        x3, x2, x1 = x2, x1, x
        z = mu + x
        z -= z.mean(1, keepdims=True)
        out[:, s, :] = np.clip(center + z, p["p_min"], p["p_max"])
    return out


# Calibrate the anchor weight lambda so the long-run cross-sectional SD matches the 2011–2019 data
if DYN["anchor_weight"] == "calibrate":
    probe = simulate_power(RHO_INIT, LONG_RUN, 200, 20, seed=1, anchor=0.0)
    v0 = probe[:, 100:, :].std(axis=2).mean() ** 2         # cross-sectional variance without anchoring
    lam = float(np.clip(np.sqrt(max(TARGET_SD ** 2 - v0, 0) / LONG_RUN.var()), 0, 1))
    DYN["anchor_weight"] = round(lam, 3)
    print(f"Unanchored long-run SD {np.sqrt(v0):.2f}; data {TARGET_SD:.2f}; calibrated anchor weight lambda = {DYN['anchor_weight']}")

t0 = time.time()
demo = simulate_power(RHO_INIT, LONG_RUN, 50, 20, seed=SEED)
print(f"simulate_power: 20 reps × 50 seasons × {len(TEAMS_SIM)} teams in {1000 * (time.time() - t0):.0f} ms; "
      f"min {demo.min():.1f}, max {demo.max():.1f}, long-run SD {demo[:, 25:, :].std(2).mean():.2f}")

# %% [md]
# ### 5.1 Validation
# The repository's Massey extract covers 2011–2020, so we compare like with like: 9-season windows of simulated teams versus
# the 2011–2019 histories of the real programs (2020 is excluded as a pandemic season), using the two statistics of
# Appendix C: within-team SD and lag-1 autocorrelation. If `massey_1960_2019.csv` is supplied, the paper's 60-season test is
# run as well.
#
# **Reading the result.** Year-to-year persistence is close to the recent decade (median lag-1 autocorrelation 0.31 simulated
# vs 0.34 observed; KS p ≈ 0.03). Within-team swings over 2011–2019 are about 20% larger than the model produces with the
# published parameters (σε = 4.5, calibrated by Scott on 1960–2019), and the KS test clearly rejects on that statistic. Part of
# the gap is the [35, 85] floor, which holds the weakest programs nearly constant (the spike near zero in the left panel).
# Raising σε to about 5.5 matches the recent swings but lowers the autocorrelation further. We keep the published
# parameters; set `DYN["sigma_eps"] = 5.5` to run the more volatile variant.
#
# **Missing input.** The paper's Appendix C reports KS tests on 113 programs over 1960–2019; that panel is not in the project. Supplying `massey_1960_2019.csv` (columns `Team, Year, Power`) reruns that test below.

# %%
def team_stats(panel):
    """panel: (seasons, teams) → within-team SD and lag-1 autocorrelation of the season-centered rating."""
    z = panel - panel.mean(1, keepdims=True)
    sd = z.std(0, ddof=1)
    a, b = z[:-1] - z[:-1].mean(0), z[1:] - z[1:].mean(0)
    ac = (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return sd, ac

full = power_df[(power_df.Year >= 2011) & (power_df.Year <= 2019)].pivot(index="Year", columns="Team", values="Power").dropna(axis=1)
sd_hist, ac_hist = team_stats(full.to_numpy())
sim = simulate_power(RHO_INIT, LONG_RUN, 60, 40, seed=7)
win = sim[:, -9:, :]                                        # last 9 seasons of each replication
sd_sim = np.concatenate([team_stats(w)[0] for w in win]); ac_sim = np.concatenate([team_stats(w)[1] for w in win])
ks_sd, ks_ac = stats.ks_2samp(sd_hist, sd_sim), stats.ks_2samp(ac_hist, ac_sim)
val = pd.DataFrame({"statistic": ["within-team SD", "lag-1 autocorrelation"],
                    "data median": [np.median(sd_hist), np.median(ac_hist)], "simulated median": [np.median(sd_sim), np.median(ac_sim)],
                    "KS D": [ks_sd.statistic, ks_ac.statistic], "p-value": [ks_sd.pvalue, ks_ac.pvalue]})
print(f"Real programs with complete 2011–2019 ratings: {full.shape[1]}; simulated team-windows: {len(sd_sim)}")
print(val.round(3).to_string(index=False))

mh = find_file("massey_hist", required=False)
VAL60 = None
if mh is not None:
    # 1960-2019 panel: programs rated in every available season. The supplied file has no 1991 season, so the lag-1
    # autocorrelation uses consecutive-season pairs only (1990 -> 1992 is skipped).
    raw = pd.read_csv(mh)
    mp = raw.pivot_table(index="Year", columns="Team", values="Power").loc[1960:2019]
    mp = mp.dropna(axis=1)
    yrs = mp.index.to_numpy()
    print(f"\nHistorical panel: {mp.shape[1]} programs rated in all {len(yrs)} available seasons 1960–2019"
          f"{' (missing: ' + ', '.join(str(y) for y in sorted(set(range(1960, 2020)) - set(yrs))) + ')' if len(yrs) < 60 else ''}")

    def team_stats_gaps(panel, years):
        z = panel - panel.mean(1, keepdims=True)
        sd = z.std(0, ddof=1)
        ok = np.diff(years) == 1
        a, b = z[:-1][ok], z[1:][ok]
        a, b = a - a.mean(0), b - b.mean(0)
        return sd, (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))

    sd_h60, ac_h60 = team_stats_gaps(mp.to_numpy(), yrs)
    rows60 = []
    for lab, anc in [("paper (era levels around 0)", 0.0), (f"notebook (anchored, λ={DYN['anchor_weight']})", None)]:
        sim60 = simulate_power(RHO_INIT, LONG_RUN, 60, 10, seed=11, anchor=anc)[:, :, :]
        sims = [team_stats_gaps(w[np.r_[0:31, 32:60]], yrs) for w in sim60]          # same 59-season pattern
        sd_s = np.concatenate([x[0] for x in sims]); ac_s = np.concatenate([x[1] for x in sims])
        k1, k2 = stats.ks_2samp(sd_h60 ** 2, sd_s ** 2), stats.ks_2samp(ac_h60, ac_s)
        rows60.append({"model": lab, "variance: data median": np.median(sd_h60 ** 2), "variance: sim median": np.median(sd_s ** 2),
                       "KS D (var)": k1.statistic, "p (var)": k1.pvalue, "AC: data median": np.median(ac_h60),
                       "AC: sim median": np.median(ac_s), "KS D (AC)": k2.statistic, "p (AC)": k2.pvalue})
    VAL60 = pd.DataFrame(rows60)
    print("Appendix C test (59 seasons, rating variance and lag-1 autocorrelation; m = 1,340 simulated team histories):")
    print(VAL60.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, h, s_, lab in [(axes[0], sd_hist, sd_sim, "within-team SD (9 seasons)"), (axes[1], ac_hist, ac_sim, "lag-1 autocorrelation")]:
    ax.hist(h, bins=20, density=True, alpha=.55, label="Massey 2011–2019")
    ax.hist(s_, bins=40, density=True, alpha=.55, label="simulated")
    ax.set_xlabel(lab); ax.legend()
for t, col in [("ALA", "crimson"), ("Ohio State", "gray"), ("Indiana", "darkred"), ("Nebraska", "red"), ("Boise", "navy"), ("UMASS", "green")]:
    axes[2].plot(demo[0, :, TEAMS_SIM.index(t)], label=t, color=col, lw=1.4)
axes[2].set_ylim(30, 90); axes[2].set_xlabel("season"); axes[2].set_ylabel("power rating"); axes[2].set_title("one simulated future")
axes[2].legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(OUT_DIR / "dynamics_validation.png", dpi=150); plt.show()
