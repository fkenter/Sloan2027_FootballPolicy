# %% [md]
# ### 7.1 FBS-wide tiers with a points table (fluid promotion and relegation)
#
# Strict promotion and relegation moves a fixed number of teams across every tier boundary each season. The **points**
# version replaces that rule with a league table that carries memory:
#
# 1. **Season points.** After each season every team earns
#    $$\text{points}_i = w_{\text{win}}\cdot\text{wins}_i \;+\; w_{\text{level}}\cdot(T-\text{tier}_i) \;+\; w_{\text{place}}\cdot(n_{\text{tier}}-\text{place}_i),$$
#    where tier 1 is the top tier (so the top tier earns $w_{\text{level}}(T-1)$ level points and the bottom tier none),
#    and place 1 is the team that finished first in its tier by wins, then point differential (so the tier winner earns
#    $w_{\text{place}}(n_{\text{tier}}-1)$ place points and the last-placed team none). One win is worth one point.
# 2. **Weighted total.** Each team's standing is the weighted average of its points over the last four seasons,
#    $S_i = \sum_{k=0}^{3}\omega_k\,\text{points}_{i,t-k}\big/\sum_k \omega_k$ (fewer seasons in the first years).
# 3. **Repositioning.** All teams are re-ranked by $S_i$ (ties broken by the results-based rating) and cut into tiers of
#    the same sizes as before. Any number of teams can move, and a team can jump more than one tier.
#
# Level points make tiers sticky (a team must outscore the teams above it *including* the bonus they earn for playing in a
# higher tier); place points reward finishing high within one's tier; the season weights set how much one season's luck
# matters. $w_{\text{level}} = 0$ with only last season's points is a pure re-sort; a very large $w_{\text{level}}$ freezes
# the tiers.

# %%
WEIGHT_PROFILES = {
    "last season only": (1.0, 0.0, 0.0, 0.0),
    "steep (1, 1/2, 1/4, 1/8)": (1.0, 0.5, 0.25, 0.125),
    "moderate (1, 3/4, 1/2, 1/4)": (1.0, 0.75, 0.5, 0.25),
    "equal (1, 1, 1, 1)": (1.0, 1.0, 1.0, 1.0),
}


class FBSPointsTiers(FBSTiers):
    """FBS-wide tiers repositioned each season by a weighted four-season points total (wins, tier level, tier place)."""
    name, key = "FBS-wide tiers (points)", "fbs_points"
    params = {"n": 12, "level": 4.0, "place": 0.5, "weights": "moderate (1, 3/4, 1/2, 1/4)", "fill": "random"}

    def seed(self):
        super().seed()
        self.history = []

    def season_points(self, res):
        T = self.n_tiers
        pts = self.params.get("win", 1.0) * res["wins"] + self.params["level"] * (T - 1 - self.tier)
        for b in self.blocks:
            ranked = standings(b, res)
            pts[ranked] += self.params["place"] * (len(b) - 1 - np.arange(len(b)))
        return pts

    def update(self, pairs, margin):
        Policy.update(self, pairs, margin)                  # results-based rating (tie-breaker)
        res = season_results(pairs, margin, self.ctx.N)
        self.history.insert(0, self.season_points(res))
        self.history = self.history[:4]
        w = np.array(WEIGHT_PROFILES[self.params["weights"]][:len(self.history)])
        if w.sum() == 0:
            w = np.ones(len(self.history))
        S = sum(wk * h for wk, h in zip(w, self.history)) / w.sum()
        self.order = np.argsort(-(S + 1e-6 * self.rating), kind="stable")
        self._tiers()


POLICY_CLASSES["fbs_points"] = FBSPointsTiers


class FBSPointsChamp(FBSPointsTiers):
    """Points table plus a bonus for finishing first in one's tier ("winning your league")."""
    name, key = "FBS-wide tiers (points + tier-winner bonus)", "fbs_champ"
    params = {"n": 12, "level": 6.0, "place": 1.0, "weights": "steep (1, 1/2, 1/4, 1/8)", "fill": "random", "champ": 0.0}

    def season_points(self, res):
        pts = super().season_points(res)
        self._winners = [standings(b, res)[0] for b in self.blocks]
        for w in self._winners:
            pts[w] += self.params["champ"]
        return pts

    def update(self, pairs, margin):
        old_tier = self.tier.copy()
        super().update(pairs, margin)
        movers = self._winners[1:]                          # winners of every tier below the top
        self.last_stats = {"winner_promoted": float(np.mean([self.tier[t] < old_tier[t] for t in movers])),
                           "winner_jump2": float(np.mean([old_tier[t] - self.tier[t] >= 2 for t in movers]))}


POLICY_CLASSES["fbs_champ"] = FBSPointsChamp
