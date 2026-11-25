# %% [md]
# ## 6. League policies, written as rules a league office could run
#
# Each policy decides **who plays whom** (every team plays exactly 12 FBS games). Which of those games are televised,
# and where, is decided separately (Section 7), so the policy and the TV decision can be measured apart.
#
# All movement between tiers uses **results only**: standings (wins, then point differential) within a tier or division,
# or, where a single ordered list of all teams is needed, a *results-based rating* computed from last season's scores
# (a least-squares point-margin rating of the kind Massey publishes). True team strength never enters a policy decision.
# Before season 1, every policy is seeded from one *warm-up* season played under the status quo.
#
# | Policy | Season structure | End-of-season movement | Tuned settings |
# |---|---|---|---|
# | (i) **Status quo** | 8–9 conference games (SEC, Big Ten, Big 12 play 9 from 2026), remaining games non-conference | none | — |
# | (ii) **Like-opponent games** | Conference games as now, then `k` non-conference games against the nearest-rated teams from other conferences, rest non-conference | ratings recomputed from the season's scores | `k` |
# | (iii) **Conference tiers** | Each conference split into an upper and a lower division; round robin within the division, `c` crossover games, rest non-conference | bottom `m` of each upper division swap with top `m` of the lower division | `c`, `m` |
# | (iv) **FBS-wide tiers** | All of FBS in tiers of about `n` teams; round robin within the tier, remaining games against teams from neighbouring tiers (`fill`: nearest in the ranking, or random from the tier above or below) | top `p` of each tier promoted, bottom `p` relegated (or `p = "re-sort"`: full re-sort on the results-based rating) | `n`, `p`, `fill` |
#
# This version has **no conference brand effects**: the audience model was re-estimated without conference terms, so
# conference membership affects only who plays whom.

# %%
SIM = CONFIG["simulation"]
SEASON_GAMES = SIM["season_games"]
SIGMA_GAME = SIM["sigma_game"]
CONF_GAMES = SIM["conference_games"]


class LeagueContext:
    def __init__(self, teams, conf_of):
        self.teams = list(teams)
        self.N = len(teams)
        self.conf_names = list(dict.fromkeys(conf_of[t] for t in teams))
        self.conf = np.array([self.conf_names.index(conf_of[t]) for t in teams])
        self.members = [np.nonzero(self.conf == c)[0] for c in range(len(self.conf_names))]


def rr_rounds(members, rng):
    """Randomized single round robin (circle method): list of rounds, each a list of (i, j)."""
    m = list(rng.permutation(members))
    if len(m) % 2:
        m.append(-1)
    n = len(m)
    rounds = []
    for _ in range(n - 1):
        rounds.append([(m[k], m[n - 1 - k]) for k in range(n // 2) if m[k] >= 0 and m[n - 1 - k] >= 0])
        m = [m[0]] + [m[-1]] + m[1:-1]
    rng.shuffle(rounds)
    return rounds


class Fixtures:
    """A season's matchups, with O(1) duplicate and game-count checks and a completion step that guarantees every
    team its full schedule whenever that is possible."""

    def __init__(self, N):
        self.N = N
        self.played = np.zeros((N, N), bool)
        self.count = np.zeros(N, int)
        self.pairs = []

    def add(self, i, j):
        if i == j or self.played[i, j] or self.count[i] >= SEASON_GAMES or self.count[j] >= SEASON_GAMES:
            return False
        self.played[i, j] = self.played[j, i] = True
        self.count[i] += 1; self.count[j] += 1
        self.pairs.append((min(i, j), max(i, j)))
        return True

    def remove(self, i, j):
        self.played[i, j] = self.played[j, i] = False
        self.count[i] -= 1; self.count[j] -= 1
        self.pairs.remove((min(i, j), max(i, j)))

    def fill(self, target, rng, allowed=None, rank=None):
        """Give every team up to `target` games. allowed(t) -> bool mask of acceptable partners for t.
        rank=None: random partner; otherwise the partner nearest in `rank` (lower = better)."""
        order = rng.permutation(self.N) if rank is None else np.argsort(rank, kind="stable")
        for t in order:
            while self.count[t] < target:
                ok = (self.count < target) & ~self.played[t]
                if allowed is not None:
                    ok &= allowed(t)
                ok[t] = False
                cand = np.nonzero(ok)[0]
                if len(cand) == 0:
                    break
                j = rng.choice(cand) if rank is None else cand[np.argmin(np.abs(rank[cand] - rank[t]) + 1e-6 * rng.random(len(cand)))]
                self.add(t, j)
        return self

    def complete(self, rng, target=SEASON_GAMES):
        """Bring every team to `target` games: first any unplayed partner; then repairs that break one existing game
        (a, b) and replace it with (t, a) and (t, b) when team t is two or more games short, or with (t, a) and (u, b)
        for two short teams t and u. Only an odd total can leave one team a single game short."""
        self.fill(target, rng)
        for _ in range(6 * self.N):
            deficit = target - self.count
            short = np.nonzero(deficit > 0)[0]
            if len(short) == 0:
                break
            t = short[np.argmax(deficit[short])]
            u = t if deficit[t] >= 2 else next((x for x in short if x != t), None)
            if u is None:
                break
            done = False
            for (a, b) in rng.permutation(np.array(self.pairs)):
                a, b = int(a), int(b)
                if t in (a, b) or u in (a, b):
                    continue
                for x, y in ((a, b), (b, a)):
                    if not self.played[t, x] and not self.played[u, y]:
                        self.remove(a, b); self.add(t, x); self.add(u, y); done = True
                        break
                if done:
                    break
            if not done:
                break
        return self

    def array(self):
        return np.array(self.pairs, int)


def conference_games(fx, ctx, rng):
    for c, mem in enumerate(ctx.members):
        k = min(CONF_GAMES.get(ctx.conf_names[c], 8), len(mem) - 1)
        for rnd in rr_rounds(mem, rng)[:k]:
            for i, j in rnd:
                fx.add(i, j)


def season_results(pairs, margin, N):
    i, j = pairs[:, 0], pairs[:, 1]
    games = np.bincount(i, minlength=N) + np.bincount(j, minlength=N)
    wins = np.bincount(i, weights=margin > 0, minlength=N) + np.bincount(j, weights=margin < 0, minlength=N)
    diff = np.bincount(i, weights=margin, minlength=N) - np.bincount(j, weights=margin, minlength=N)
    return dict(games=games, wins=wins, diff=diff, wp=wins / np.maximum(games, 1))


def standings(idx, res):
    """Teams in `idx` ordered first to last by wins, then point differential."""
    key = res["wins"][idx] + 1e-4 * res["diff"][idx]
    return np.asarray(idx)[np.argsort(-key, kind="stable")]


def results_rating(pairs, margin, N, ridge=2.0):
    """Least-squares point-margin rating from one season's scores (ridge-regularized; mean zero)."""
    X = sparse.csr_matrix((np.r_[np.ones(len(pairs)), -np.ones(len(pairs))],
                           (np.r_[np.arange(len(pairs)), np.arange(len(pairs))], np.r_[pairs[:, 0], pairs[:, 1]])),
                          shape=(len(pairs), N))
    A = (X.T @ X).toarray() + ridge * np.eye(N)
    r = np.linalg.solve(A, X.T @ np.clip(margin, -35, 35))     # margins capped at 35, as most computer ratings do
    return r - r.mean()


class Policy:
    name = "base"
    key = "base"
    params = {}

    def __init__(self, ctx, **params):
        self.ctx = ctx
        self.params = {**self.params, **params}

    def setup(self, warm_pairs, warm_margin):
        """Seed from a warm-up status-quo season: results-based rating over all FBS."""
        self.rating = results_rating(warm_pairs, warm_margin, self.ctx.N)
        self.seed()

    def seed(self):
        pass

    def update(self, pairs, margin):
        self.rating = results_rating(pairs, margin, self.ctx.N)

    def groups(self):
        return self.ctx.conf

    def label(self):
        return self.name + (" (" + ", ".join(f"{k}={v}" for k, v in self.params.items()) + ")" if self.params else "")


class StatusQuo(Policy):
    name, key = "Status quo", "sq"

    def schedule(self, rng):
        fx = Fixtures(self.ctx.N)
        conference_games(fx, self.ctx, rng)
        conf = self.ctx.conf
        fx.fill(SEASON_GAMES, rng, allowed=lambda t: conf != conf[t])
        return fx.complete(rng).array()


class LikeOpponents(Policy):
    name, key = "Like-opponent games", "like"
    params = {"k": 3}

    def schedule(self, rng):
        fx = Fixtures(self.ctx.N)
        conf = self.ctx.conf
        conference_games(fx, self.ctx, rng)
        order = np.empty(self.ctx.N); order[np.argsort(-self.rating, kind="stable")] = np.arange(self.ctx.N)
        target = np.minimum(fx.count + self.params["k"], SEASON_GAMES)
        for t in np.argsort(order):                                     # best-rated team chooses first
            while fx.count[t] < target[t]:
                ok = (fx.count < target) & ~fx.played[t] & (conf != conf[t]); ok[t] = False
                cand = np.nonzero(ok)[0]
                if len(cand) == 0:
                    break
                fx.add(t, cand[np.argmin(np.abs(order[cand] - order[t]))])
        fx.fill(SEASON_GAMES, rng, allowed=lambda t: conf != conf[t])
        return fx.complete(rng).array()


class ConferenceTiers(Policy):
    name, key = "Conference tiers", "conf_tiers"
    params = {"c": 3, "m": 2}
    MIN_SIZE = 8                                  # conferences smaller than this are not split

    def seed(self):
        self.div = {}
        for c, mem in enumerate(self.ctx.members):
            if len(mem) >= self.MIN_SIZE:
                order = mem[np.argsort(-self.rating[mem], kind="stable")]
                k = (len(order) + 1) // 2
                self.div[c] = [order[:k], order[k:]]

    def schedule(self, rng):
        fx = Fixtures(self.ctx.N)
        conf = self.ctx.conf
        for c, mem in enumerate(self.ctx.members):
            if c not in self.div:
                for rnd in rr_rounds(mem, rng)[:min(CONF_GAMES.get(self.ctx.conf_names[c], 8), len(mem) - 1)]:
                    for i, j in rnd:
                        fx.add(i, j)
                continue
            up, lo = self.div[c]
            for d in (up, lo):
                for rnd in rr_rounds(d, rng):
                    for i, j in rnd:
                        fx.add(i, j)
            cross = defaultdict(int)                               # c crossover games per team
            for _ in range(self.params["c"]):
                for t in rng.permutation(up):
                    if cross[t] >= self.params["c"]:
                        continue
                    for j in rng.permutation(lo):
                        if cross[j] < self.params["c"] and fx.add(t, j):
                            cross[t] += 1; cross[j] += 1
                            break
        fx.fill(SEASON_GAMES, rng, allowed=lambda t: conf != conf[t])
        return fx.complete(rng).array()

    def update(self, pairs, margin):
        super().update(pairs, margin)
        res = season_results(pairs, margin, self.ctx.N)
        m = self.params["m"]
        for c, (up, lo) in self.div.items():
            if m == "re-sort":
                mem = np.r_[up, lo]
                order = mem[np.argsort(-self.rating[mem], kind="stable")]
                self.div[c] = [order[:len(up)], order[len(up):]]
                continue
            su, sl = standings(up, res), standings(lo, res)
            self.div[c] = [np.r_[su[:len(su) - m], sl[:m]], np.r_[su[len(su) - m:], sl[m:]]]

    def groups(self):
        g = self.ctx.conf * 2
        for c, (up, lo) in self.div.items():
            g[lo] += 1
        return g


class FBSTiers(Policy):
    name, key = "FBS-wide tiers", "fbs_tiers"
    params = {"n": 8, "p": 2, "fill": "adjacent"}

    def seed(self):
        self.order = np.argsort(-self.rating, kind="stable")          # best to worst
        self._tiers()

    def _tiers(self):
        self.n_tiers = max(1, int(round(self.ctx.N / self.params["n"])))
        self.blocks = np.array_split(self.order, self.n_tiers)
        self.tier = np.empty(self.ctx.N, int)
        for k, b in enumerate(self.blocks):
            self.tier[b] = k
        self.pos = np.empty(self.ctx.N, float); self.pos[self.order] = np.arange(self.ctx.N)

    def schedule(self, rng):
        fx = Fixtures(self.ctx.N)
        for b in self.blocks:
            for rnd in rr_rounds(b, rng):
                for i, j in rnd:
                    fx.add(i, j)
        tier = self.tier
        if self.params["fill"] == "adjacent":
            fx.fill(SEASON_GAMES, rng, allowed=lambda t: tier != tier[t], rank=self.pos)
        else:
            fx.fill(SEASON_GAMES, rng, allowed=lambda t: np.abs(tier - tier[t]) == 1)
        return fx.complete(rng).array()

    def update(self, pairs, margin):
        super().update(pairs, margin)
        p = self.params["p"]
        if p == "re-sort":
            self.order = np.argsort(-self.rating, kind="stable")
        else:
            res = season_results(pairs, margin, self.ctx.N)
            ranked = [list(standings(b, res)) for b in self.blocks]
            for k in range(len(ranked) - 1):                          # swap bottom p of tier k with top p of tier k+1
                up, dn = ranked[k], ranked[k + 1]
                q = min(p, len(up) // 2, len(dn) // 2)
                ranked[k], ranked[k + 1] = up[:len(up) - q] + dn[:q], up[len(up) - q:] + dn[q:]
            self.order = np.array([t for r in ranked for t in r])
        self._tiers()

    def groups(self):
        return self.tier


POLICY_CLASSES = {"sq": StatusQuo, "like": LikeOpponents, "conf_tiers": ConferenceTiers, "fbs_tiers": FBSTiers}

# %% [md]
# **Structural checks.** Every policy must give every team 12 games; the rating used for movement is computed from
# scores only (its correlation with true strength after one season is reported for reference).

# %%
TEAMS_SIM = [t for g in ALIGN_2025.values() for t in g]
CONF_SIM = {t: c for c, g in ALIGN_2025.items() for t in g}
CTX = LeagueContext(TEAMS_SIM, CONF_SIM)


def game_noise(rep, season, N=CTX.N, seed=SEED):
    """Common random numbers: z[i, j] = -z[j, i] ~ N(0, 1) for every possible game of a (replication, season)."""
    rng = np.random.default_rng([seed, rep, season, 99])
    z = rng.standard_normal((N, N))
    return np.triu(z, 1) - np.triu(z, 1).T


def warmup_season(rho0, rep, seed=SEED):
    """Status-quo season played before season 1; its scores seed every policy."""
    fx = Fixtures(CTX.N)
    rng = np.random.default_rng([seed, rep, 77])
    conference_games(fx, CTX, rng)
    fx.fill(SEASON_GAMES, rng, allowed=lambda t: CTX.conf != CTX.conf[t])
    pairs = fx.complete(rng).array()
    z = game_noise(rep, 10_000)
    return pairs, rho0[pairs[:, 0]] - rho0[pairs[:, 1]] + SIGMA_GAME * z[pairs[:, 0], pairs[:, 1]]


_wp, _wm = warmup_season(RHO_INIT, 0)
_r = results_rating(_wp, _wm, CTX.N)
print(f"Results-based rating after one warm-up season vs true strength: Spearman {stats.spearmanr(_r, RHO_INIT).statistic:.2f}\n")
_rng = np.random.default_rng(0)
for cls, prm in [(StatusQuo, {}), (LikeOpponents, {"k": 3}), (ConferenceTiers, {"c": 3, "m": 2}),
                 (FBSTiers, {"n": 8, "p": 2, "fill": "adjacent"}), (FBSTiers, {"n": 12, "p": 2, "fill": "random"})]:
    P = cls(CTX, **prm); P.setup(_wp, _wm)
    for season in range(3):
        t0 = time.time(); pr = P.schedule(_rng); dt = time.time() - t0
        cnt = np.bincount(pr.ravel(), minlength=CTX.N)
        assert len({tuple(x) for x in pr}) == len(pr), "duplicate game"
        P.update(pr, RHO_INIT[pr[:, 0]] - RHO_INIT[pr[:, 1]] + SIGMA_GAME * _rng.standard_normal(len(pr)))
    same_conf = (CTX.conf[pr[:, 0]] == CTX.conf[pr[:, 1]]).mean()
    print(f"{P.label():<52} games {len(pr):4d} | games per team min/max {cnt.min()}/{cnt.max()} | "
          f"intra-conference {100 * same_conf:4.1f}% | {1000 * dt:.0f} ms")
