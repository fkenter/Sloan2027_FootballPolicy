# %% [md]
# ### 3.2 Scheduler implementation
#
# **Formulation.** Variables are unordered matchups $\{i,j\}$ × medium-specific positions $s=(w,d,t,b)$ with capacity
# $K_s>0$ (146 positions, 303 seats). The objective (A.1) uses the *linearized* value $\hat y_{ijs}$, in which the number
# of games and the total power in each window are fixed at the window's reference values; every schedule is then
# re-scored with the window composition it actually produced (**realized** viewership, the number reported throughout).
#
# **Solution pipeline** (Appendix B):
# 1. *Safe pruning* of positions with no capacity, of games whose own power exceeds the window cap and, when games are
#    pinned to weeks, of positions outside the game's week.
# 2. *Randomized greedy construction* (GRASP variant): rank all (matchup, position) candidates by
#    $\hat y\cdot e^{\tau\varepsilon}$ and accept each one that keeps every active constraint satisfied (with a look-ahead
#    that keeps enough power in a window for its remaining seats).
# 3. *Local search*: one-seat neighborhoods (empty a seat and refill it with its best feasible matchup) and two-seat swaps.
# 4. *Exposure repair* when `min_exposure` > 0.
# 5. *Optional MIP polish* on a top-K candidate pool (Gurobi if installed, else HiGHS).
# 6. **Exact solution for pinned games** (new): when every game is pinned to its week (and the rest rule, the only
#    constraint linking weeks, is off) the problem splits into one small assignment MIP per week. With σ = 1 and a fixed
#    set of games, each window's power must be matched almost exactly (a subset-sum problem), so the MIPs allow a
#    rounding tolerance of 0.5 rating points per window (≈0.1% of a window's power) and a 2-second limit per week,
#    keeping the incumbent if the limit is reached.
#
# Every schedule is passed through an independent `audit()` that rechecks all active constraints.
#
# ### 3.3 The "2019-style" television rule (no optimization)
# To separate *optimization* from *policy* we need a way to televise any season's games **the way networks did in
# 2019**, without optimizing. Score every game by its *matchup appeal* (the viewership model's matchup terms: game power,
# power gap, brands, rivalry, at a typical window). In 2019 networks televised the games at certain appeal ranks of the
# season (the 1st, 2nd, 5th, 40th … most appealing) and put each in a certain slot. The rule does the same in any
# season: the game at the same appeal quantile of this season's schedule goes into the same slot (the nearest free game
# if its teams are already televised that week). Applied to the 2019 schedule, the rule reproduces the 2019 slate
# exactly; it keeps 2019's *selectivity* (how far down the list networks reached) as well as its slotting.

# %%
CAP_TOL = 0.5    # rounding tolerance (rating points) on window power caps in the exact MIPs; see Section 4.1

class Inventory:
    """Broadcast windows (week, day, time) and medium-specific positions from the 2019 inventory file."""

    def __init__(self, df, vm, calendar):
        self.df = df.reset_index(drop=True)
        self.week = df["Week"].to_numpy()
        self.day = df["NumDay"].to_numpy()
        self.time = np.array([TIMES.index(t) for t in df["Slot"]])
        self.G = df["Num Games"].to_numpy().astype(float)
        self.Ptot = df["Total Power"].to_numpy().astype(float)
        self.weeks = np.sort(np.unique(self.week))
        self.week_idx = np.searchsorted(self.weeks, self.week)
        self.nW = len(self.weeks)
        self.wconst = np.array([vm.window_const(w, d, TIMES[t]) for w, d, t in zip(self.week, self.day, self.time)])
        late, early = set(calendar["late_days"]), set(calendar["early_days"])
        rows = []
        for r in range(len(df)):
            for m, col in enumerate(["Num Broadcast", "Num Cable"]):
                cap = int(df.at[r, col])
                if cap > 0:
                    rows.append((r, m, cap))
        self.pos_win = np.array([r for r, _, _ in rows])
        self.pos_med = np.array([m for _, m, _ in rows])
        self.pos_cap = np.array([c for _, _, c in rows])
        self.pos_week = self.week_idx[self.pos_win]
        self.pos_time = self.time[self.pos_win]
        self.pos_late = np.isin(self.day[self.pos_win], list(late))
        self.pos_early = np.isin(self.day[self.pos_win], list(early))
        self.mconst = np.array([vm.medium[MEDIA[m]] for m in self.pos_med])
        self.S = len(rows)
        self.n_seats = int(self.pos_cap.sum())

    def describe(self):
        return (f"{len(self.df)} windows, {self.S} medium-specific positions, {self.n_seats} seats, "
                f"{self.nW} weeks ({self.weeks.min()}–{self.weeks.max()})")


INV = Inventory(inventory_df, VM, CONFIG["calendar"])
print(INV.describe())


@dataclass
class ScheduleResult:
    table: pd.DataFrame
    linear: float
    realized: float
    stats: dict


class SeasonScheduler:
    """Config-driven season scheduler. One instance per (team set, constraint set); call `solve()` per season."""

    def __init__(self, teams, conf_of, constraints, solver=None, inv=INV, vm=VM, rivalries=RIVALRIES, config=CONFIG):
        self.teams = list(teams)
        self.N = len(self.teams)
        self.tid = {t: k for k, t in enumerate(self.teams)}
        self.conf_of = conf_of
        self.c = dict(constraints)
        self.sv = dict(config["solver"], **(solver or {}))
        self.inv, self.vm = inv, vm
        self.rivalries = rivalries
        self.tol = 1e-9                   # power-cap tolerance used by the feasibility checks
        sig = self.c.get("power_cap_slack")
        self.relin_iters = 0 if sig == 1.0 else 2       # re-linearize when the cap does not pin window composition
        # brand effects
        self.brand = np.array([vm.brand(t, conf_of.get(t, ""), self.c.get("brand_mode", "team")) for t in self.teams])
        if self.c.get("brand_mode") == "average":
            base = np.array([vm.brand(t, conf_of.get(t, ""), "team") for t in self.teams])
            self.brand = np.full(self.N, base.mean())
        # time zones
        tz = config["time_zones"]
        zone_names = [z for z in tz if not z.startswith("_")] + [tz.get("_default", "East")]
        self.zone_names = list(dict.fromkeys(zone_names))
        zmap = {t: z for z, ts in tz.items() if not z.startswith("_") for t in ts}
        self.zone = np.array([self.zone_names.index(zmap.get(t, tz.get("_default", "East"))) for t in self.teams])
        self._build_quota_rules(config)
        self._build_zone_caps(config)

    # ---------------- constraint pre-processing ----------------
    def _build_quota_rules(self, config):
        name = self.c.get("team_group_caps")
        self.q_T = self.q_O = self.q_max = None
        if not name:
            return
        spec = config["team_group_cap_sets"][name]
        groups = config["groups"][spec["groups"]]
        T, O, M = [], [], []
        for r in spec["rules"]:
            tset = set(groups[r["team_group"]])
            if "opp_group" in r:
                oset = set(groups[r["opp_group"]])
                o = np.array([t in oset for t in self.teams])
            else:
                excl = set(groups[r["opp_not_group"]])
                o = np.array([t not in excl for t in self.teams])
            T.append(np.array([t in tset for t in self.teams]))
            O.append(o)
            M.append(r["max"])
        self.q_T, self.q_O, self.q_max = np.array(T), np.array(O), np.array(M)

    def _build_zone_caps(self, config):
        name = self.c.get("time_zone_caps")
        self.z_max = np.full((len(self.zone_names), len(TIMES)), np.inf)
        if name:
            for r in config["time_zone_cap_sets"][name]:
                self.z_max[self.zone_names.index(r["zone"]), TIMES.index(r["time"])] = r["max"]
        self.z_active = np.isfinite(self.z_max)

    # ---------------- value matrix and candidates ----------------
    def _prepare(self, rho, pairs, pair_week, pair_day=None, pair_medium=None, ref_power=None, lin_power=None):
        inv, vm, c = self.inv, self.vm, self.c
        # reference total power per window: base of the cap (A.8) and, unless lin_power is given, linearization point of (A.1)
        self.ref_power = inv.Ptot.copy() if ref_power is None else np.asarray(ref_power, float)
        self.lin_power = self.ref_power if lin_power is None else np.asarray(lin_power, float)
        pi, pj = pairs[:, 0], pairs[:, 1]
        self.pi, self.pj, self.M = pi, pj, len(pi)
        self.P = rho[pi] + rho[pj]
        self.D = np.abs(rho[pi] - rho[pj])
        names = self.teams
        self.R = np.array([frozenset((names[a], names[b])) in self.rivalries for a, b in zip(pi, pj)], dtype=float)
        self.B = self.brand[pi] + self.brand[pj]
        L = vm.log_view(self.P[:, None], self.D[:, None], self.B[:, None], self.R[:, None],
                        inv.G[None, :], self.lin_power[None, :], inv.wconst[None, :], 0.0)           # M x windows
        V = np.exp(L[:, inv.pos_win] + inv.mconst[None, :])                                    # M x positions
        sigma = c.get("power_cap_slack")
        self.win_cap = self.ref_power * sigma if sigma else np.full(len(inv.Ptot), np.inf)
        mask = (self.ref_power[inv.pos_win] >= c.get("low_power_window", 0))[None, :] & np.ones((self.M, 1), bool)
        mask &= self.P[:, None] <= self.win_cap[inv.pos_win][None, :] + 1e-9                   # safe pruning
        if c["matchup_mode"] in ("fixed_week", "historical", "subset_week"):
            mask &= np.searchsorted(inv.weeks, pair_week)[:, None] == inv.pos_week[None, :]
        if c["matchup_mode"] == "historical":          # also pin day and medium: recovers the actual 2019 slate
            mask &= np.asarray(pair_day)[:, None] == inv.day[inv.pos_win][None, :]
            mask &= np.asarray(pair_medium)[:, None] == inv.pos_med[None, :]
        V[~mask] = 0.0
        self.V = V
        cm, cs = np.nonzero(mask)
        self.cm, self.cs, self.cv = cm, cs, V[cm, cs]
        self.win_seats = np.bincount(inv.pos_win, weights=inv.pos_cap, minlength=len(inv.G))
        # Lookahead for (A.4): a game may enter a window only if the power left still fits the window's other open seats,
        # each filled by the weakest teams not yet playing that week.
        self.rho = rho
        self.lookahead = bool(c.get("fill_all_seats", True))
        self.weak_order = np.argsort(rho, kind="stable")
        self.fill_margin = float(self.sv.get("fill_margin", 6.0))    # extra power kept per open seat
        order = np.argsort(cs, kind="stable")
        self.by_pos_m = np.split(cm[order], np.cumsum(np.bincount(cs, minlength=inv.S))[:-1])
        # per-pair quota increments and zone involvement
        if self.q_T is not None:
            self.q_inc_i = self.q_T[:, pi] & self.q_O[:, pj]      # rules x M : rule counts for team i
            self.q_inc_j = self.q_T[:, pj] & self.q_O[:, pi]
        self.pz_i, self.pz_j = self.zone[pi], self.zone[pj]

    # ---------------- state helpers (NumPy) ----------------
    def _empty_state(self):
        N, nW, inv = self.N, self.inv.nW, self.inv
        st = dict(tw=np.zeros((N, nW), bool), late=np.zeros((N, nW), bool), early=np.zeros((N, nW), bool),
                  cnt=np.zeros(N, int), used=np.zeros(self.M, bool), seat=inv.pos_cap.copy(),
                  winpow=self.win_cap.copy(), winseat=self.win_seats.copy(), zc=np.zeros_like(self.z_max), assign={s: [] for s in range(inv.S)})
        if self.q_T is not None:
            st["qc"] = np.zeros(self.q_T.shape, int)
        return st

    def _apply(self, st, m, s, sign):
        inv = self.inv
        i, j, w = self.pi[m], self.pj[m], inv.pos_week[s]
        on = sign > 0
        st["tw"][[i, j], w] = on
        if inv.pos_late[s]:
            st["late"][[i, j], w] = on
        if inv.pos_early[s]:
            st["early"][[i, j], w] = on
        st["cnt"][[i, j]] += sign
        st["used"][m] = on
        st["seat"][s] -= sign
        st["winpow"][inv.pos_win[s]] -= sign * self.P[m]
        st["winseat"][inv.pos_win[s]] -= sign
        t = inv.pos_time[s]
        for z in {self.pz_i[m], self.pz_j[m]}:
            st["zc"][z, t] += sign
        if self.q_T is not None:
            st["qc"][:, i] += sign * self.q_inc_i[:, m]
            st["qc"][:, j] += sign * self.q_inc_j[:, m]
        if on:
            st["assign"][s].append(m)
        else:
            st["assign"][s].remove(m)

    def _feasible(self, st, s, ms, lookahead=True):
        """Vectorized check: which matchups in `ms` can be placed in position `s` given state `st`."""
        inv, c = self.inv, self.c
        i, j, w = self.pi[ms], self.pj[ms], inv.pos_week[s]
        ok = (~st["used"][ms]) & (st["cnt"][i] < c["max_games_per_team"]) & (st["cnt"][j] < c["max_games_per_team"])
        win = inv.pos_win[s]
        k = int(st["winseat"][win]) - 1
        reserve = 0.0
        if lookahead and self.lookahead and k > 0:
            free_r = self.rho[self.weak_order][~st["tw"][self.weak_order, w]]
            reserve = free_r[: 2 * k].sum() + k * self.fill_margin
        ok &= self.P[ms] <= st["winpow"][win] - reserve + self.tol
        if c.get("one_game_per_week", True):
            ok &= ~st["tw"][i, w] & ~st["tw"][j, w]
        if c.get("rest_rule", True):
            if inv.pos_late[s] and w + 1 < inv.nW:
                ok &= ~st["early"][i, w + 1] & ~st["early"][j, w + 1]
            if inv.pos_early[s] and w > 0:
                ok &= ~st["late"][i, w - 1] & ~st["late"][j, w - 1]
        if self.q_T is not None:
            room = st["qc"] < self.q_max[:, None]                      # rules x N
            ok &= ~(self.q_inc_i[:, ms] & ~room[:, i]).any(0)
            ok &= ~(self.q_inc_j[:, ms] & ~room[:, j]).any(0)
        t = inv.pos_time[s]
        if self.z_active[:, t].any():
            full = st["zc"][:, t] >= self.z_max[:, t]
            ok &= ~full[self.pz_i[ms]] & ~full[self.pz_j[ms]]
        return ok

    # ---------------- construction ----------------
    def _greedy(self, order):
        """Fast scalar greedy over ranked candidates (Python lists for speed)."""
        inv, c = self.inv, self.c
        N, nW = self.N, inv.nW
        cm, cs = self.cm[order].tolist(), self.cs[order].tolist()
        pi, pj, P = self.pi.tolist(), self.pj.tolist(), self.P.tolist()
        pw, pwin = inv.pos_week.tolist(), inv.pos_win.tolist()
        plate, pearly, ptime = inv.pos_late.tolist(), inv.pos_early.tolist(), inv.pos_time.tolist()
        seat, winpow = inv.pos_cap.tolist(), self.win_cap.tolist()
        winseat, look = self.win_seats.tolist(), self.lookahead
        rho_l, weak, margin = self.rho.tolist(), self.weak_order.tolist(), self.fill_margin

        def reserve(w, k, i, j):
            tot, need = 0.0, 2 * k
            for t in weak:
                if need == 0:
                    break
                if t == i or t == j or tw[t * nW + w]:
                    continue
                tot += rho_l[t]
                need -= 1
            return tot + k * margin

        used, tw = bytearray(self.M), bytearray(N * nW)
        late, early = bytearray(N * nW), bytearray(N * nW)
        cnt, maxg = [0] * N, c["max_games_per_team"]
        rest, weekly = c.get("rest_rule", True), c.get("one_game_per_week", True)
        quota = self.q_T is not None
        if quota:
            qmax = self.q_max.tolist()
            qc = [[0] * N for _ in qmax]
            inc_i = [np.nonzero(col)[0].tolist() for col in self.q_inc_i.T]
            inc_j = [np.nonzero(col)[0].tolist() for col in self.q_inc_j.T]
        zmax = self.z_max.tolist()
        zc = [[0] * len(TIMES) for _ in zmax]
        zactive = bool(self.z_active.any())
        pzi, pzj = self.pz_i.tolist(), self.pz_j.tolist()
        left = sum(seat)
        out = []
        for m, s in zip(cm, cs):
            if used[m] or not seat[s]:
                continue
            i, j, w = pi[m], pj[m], pw[s]
            a, b = i * nW + w, j * nW + w
            if weekly and (tw[a] or tw[b]):
                continue
            if cnt[i] >= maxg or cnt[j] >= maxg:
                continue
            win = pwin[s]
            if P[m] > winpow[win] + 1e-9:
                continue
            if look and winseat[win] > 1 and P[m] > winpow[win] - reserve(w, winseat[win] - 1, i, j) + 1e-9:
                continue
            if rest:
                if plate[s] and w + 1 < nW and (early[a + 1] or early[b + 1]):
                    continue
                if pearly[s] and w > 0 and (late[a - 1] or late[b - 1]):
                    continue
            if quota and (any(qc[r][i] >= qmax[r] for r in inc_i[m]) or any(qc[r][j] >= qmax[r] for r in inc_j[m])):
                continue
            t = ptime[s]
            if zactive and (zc[pzi[m]][t] >= zmax[pzi[m]][t] or zc[pzj[m]][t] >= zmax[pzj[m]][t]):
                continue
            # accept
            used[m] = 1; seat[s] -= 1; tw[a] = tw[b] = 1
            if plate[s]:
                late[a] = late[b] = 1
            if pearly[s]:
                early[a] = early[b] = 1
            cnt[i] += 1; cnt[j] += 1; winpow[win] -= P[m]; winseat[win] -= 1
            if quota:
                for r in inc_i[m]:
                    qc[r][i] += 1
                for r in inc_j[m]:
                    qc[r][j] += 1
            for z in {pzi[m], pzj[m]}:
                zc[z][t] += 1
            out.append((m, s))
            left -= 1
            if left == 0:
                break
        return out

    def _state_from(self, assignment):
        st = self._empty_state()
        for m, s in assignment:
            self._apply(st, m, s, +1)
        return st

    # ---------------- improvement ----------------
    def _reopt_seats(self, st, rng, passes):
        """LNS with one-seat neighborhoods: empty a seat, refill with its best feasible matchup."""
        inv, V = self.inv, self.V
        for _ in range(passes):
            improved = 0
            seats = [(s, m) for s in range(inv.S) for m in list(st["assign"][s])]
            empty = [(s, None) for s in range(inv.S) for _ in range(st["seat"][s])]
            seats += empty
            rng.shuffle(seats)
            for s, m0 in seats:
                if m0 is not None:
                    if m0 not in st["assign"][s]:
                        continue
                    self._apply(st, m0, s, -1)
                ms = self.by_pos_m[s]
                if len(ms) == 0:
                    if m0 is not None:
                        self._apply(st, m0, s, +1)
                    continue
                ok = self._feasible(st, s, ms)
                if not ok.any():
                    if m0 is not None:
                        self._apply(st, m0, s, +1)
                    continue
                vals = np.where(ok, V[ms, s], -1.0)
                k = int(np.argmax(vals))
                old = V[m0, s] if m0 is not None else 0.0
                if vals[k] > old + 1e-9:
                    self._apply(st, ms[k], s, +1)
                    improved += 1
                elif m0 is not None:
                    self._apply(st, m0, s, +1)
            if not improved:
                break
        return st

    def _swap_moves(self, st, scope="week", max_passes=5):
        """Two-seat neighborhood: exchange the games in two positions (same week, or any week if scope='all').
        Deltas are recomputed each pass; within a pass each game moves at most once, so deltas stay exact."""
        inv, V = self.inv, self.V
        improved = 0
        groups = [np.arange(inv.S)] if scope == "all" else [np.nonzero(inv.pos_week == w)[0] for w in range(inv.nW)]
        for _ in range(max_passes):
            moved_this_pass = 0
            for grp in groups:
                items = [(s, m) for s in grp for m in st["assign"][s]]
                if len(items) < 2:
                    continue
                S_ = np.array([s for s, _ in items]); M_ = np.array([m for _, m in items])
                cur = V[M_, S_]
                cross = V[M_[:, None], S_[None, :]]                       # game a placed in the seat of b
                delta = cross + cross.T - cur[:, None] - cur[None, :]
                valid = (cross > 0) & (cross.T > 0) & (S_[:, None] != S_[None, :])
                a_idx, b_idx = np.nonzero(np.triu(valid & (delta > 1e-9), 1))
                touched = set()
                for k in np.argsort(-delta[a_idx, b_idx]):
                    a, b = a_idx[k], b_idx[k]
                    if a in touched or b in touched:
                        continue
                    (sa, ma), (sb, mb) = items[a], items[b]
                    self._apply(st, ma, sa, -1); self._apply(st, mb, sb, -1)
                    if self._feasible(st, sb, np.array([ma]), lookahead=False)[0]:
                        self._apply(st, ma, sb, +1)
                        if self._feasible(st, sa, np.array([mb]), lookahead=False)[0]:
                            self._apply(st, mb, sa, +1)
                            touched.update((a, b))
                            moved_this_pass += 1
                            continue
                        self._apply(st, ma, sb, -1)
                    self._apply(st, ma, sa, +1); self._apply(st, mb, sb, +1)
            improved += moved_this_pass
            if not moved_this_pass:
                break
        return improved

    def _repair_exposure(self, st, k_min):
        """Give every team at least `k_min` televised games, choosing the cheapest feasible replacement each time."""
        inv, V = self.inv, self.V
        failures = 0
        for _round in range(k_min):
            for u in np.argsort(st["cnt"], kind="stable"):
                if st["cnt"][u] >= _round + 1:
                    continue
                pairs_u = np.nonzero((self.pi == u) | (self.pj == u))[0]
                best = None
                for s in range(inv.S):
                    ms = pairs_u[V[pairs_u, s] > 0]
                    if len(ms) == 0:
                        continue
                    options = [None] * st["seat"][s] + list(st["assign"][s])
                    for m0 in options:
                        if m0 is not None:
                            a, b = self.pi[m0], self.pj[m0]
                            if (st["cnt"][a] <= k_min and a != u) or (st["cnt"][b] <= k_min and b != u):
                                continue
                            self._apply(st, m0, s, -1)
                        ok = self._feasible(st, s, ms)
                        if ok.any():
                            k = int(np.argmax(np.where(ok, V[ms, s], -1)))
                            loss = (V[m0, s] if m0 is not None else 0.0) - V[ms[k], s]
                            if best is None or loss < best[0]:
                                best = (loss, s, m0, ms[k])
                        if m0 is not None:
                            self._apply(st, m0, s, +1)
                if best is None:
                    failures += 1
                    continue
                _, s, m0, m1 = best
                if m0 is not None:
                    self._apply(st, m0, s, -1)
                self._apply(st, m1, s, +1)
        return failures

    # ---------------- MIP polish ----------------
    def _mip(self, incumbent, required):
        inv, c, sv = self.inv, self.c, self.sv
        K = sv["top_k"]
        keep = np.zeros(len(self.cm), bool)
        if c["matchup_mode"] in ("fixed_week", "fixed_season", "historical") or K is None:
            keep[:] = True
        else:
            for s in range(inv.S):
                idx = np.nonzero(self.cs == s)[0]
                if len(idx):
                    keep[idx[np.argsort(-self.cv[idx])[:K]]] = True
        inc_set = set(incumbent)
        keep |= np.array([(m, s) in inc_set for m, s in zip(self.cm, self.cs)])
        vm_, vs_ = self.cm[keep], self.cs[keep]
        n = len(vm_)
        val = self.V[vm_, vs_]
        A_blocks, LB, UB = [], [], []

        def block(row_of_var, nrows, lo, hi, weight=None, var_idx=None):
            var_idx = np.arange(n) if var_idx is None else var_idx
            w_ = np.ones(len(var_idx)) if weight is None else weight
            A_blocks.append(sparse.csr_matrix((w_, (row_of_var, var_idx)), shape=(nrows, n)))
            LB.append(np.broadcast_to(lo, nrows).astype(float)); UB.append(np.broadcast_to(hi, nrows).astype(float))

        i_, j_, w_ = self.pi[vm_], self.pj[vm_], inv.pos_week[vs_]
        nW, N = inv.nW, self.N
        fill = c.get("fill_all_seats", True) and c["matchup_mode"] in ("free", "subset", "subset_week")
        block(vs_, inv.S, inv.pos_cap if fill else 0, inv.pos_cap)                          # (A.7) seats, (A.4) if filling
        req = required if required is not None else np.zeros(self.M, bool)
        block(vm_, self.M, req.astype(float), 1.0)                                         # (A.2)/(A.10)/(A.11)
        if c.get("one_game_per_week", True):                                              # (A.3)
            block(np.r_[i_ * nW + w_, j_ * nW + w_], N * nW, 0, 1, var_idx=np.r_[np.arange(n), np.arange(n)])
        k_min = c.get("min_exposure", 0)
        block(np.r_[i_, j_], N, k_min, c["max_games_per_team"], var_idx=np.r_[np.arange(n), np.arange(n)])  # (A.5),(A.13)
        if c.get("rest_rule", True):                                                       # (A.6)
            lt, ea = inv.pos_late[vs_], inv.pos_early[vs_]
            rr, vv = [], []
            for team in (i_, j_):
                sel = lt & (w_ < nW - 1); rr.append(team[sel] * (nW - 1) + w_[sel]); vv.append(np.nonzero(sel)[0])
                sel = ea & (w_ > 0); rr.append(team[sel] * (nW - 1) + w_[sel] - 1); vv.append(np.nonzero(sel)[0])
            block(np.concatenate(rr), N * (nW - 1), 0, 1, var_idx=np.concatenate(vv))
        if np.isfinite(self.win_cap).any():                                                # (A.8)
            block(inv.pos_win[vs_], len(inv.Ptot), 0, np.where(np.isfinite(self.win_cap), self.win_cap, 1e9), weight=self.P[vm_])
        if self.q_T is not None:                                                           # (A.12) per-team caps
            R_ = len(self.q_max)
            rr, vv = [], []
            for inc, team in ((self.q_inc_i[:, vm_], i_), (self.q_inc_j[:, vm_], j_)):
                r_idx, v_idx = np.nonzero(inc)
                rr.append(r_idx * N + team[v_idx]); vv.append(v_idx)
            block(np.concatenate(rr), R_ * N, 0, np.repeat(self.q_max, N), var_idx=np.concatenate(vv))
        if self.z_active.any():                                                            # (A.14)
            nT = len(TIMES); rr, vv = [], []
            t_ = inv.pos_time[vs_]
            zi, zj = self.pz_i[vm_], self.pz_j[vm_]
            rr += [zi * nT + t_, (zj * nT + t_)[zj != zi]]; vv += [np.arange(n), np.nonzero(zj != zi)[0]]
            zcap = np.where(self.z_active, self.z_max, 1e9).ravel()
            block(np.concatenate(rr), zcap.size, 0, zcap, var_idx=np.concatenate(vv))
        A = sparse.vstack(A_blocks).tocsr()
        lo, hi = np.concatenate(LB), np.concatenate(UB)
        x0 = np.array([(m, s) in inc_set for m, s in zip(vm_, vs_)], float)
        t0 = time.time()
        backend = sv["mip_backend"]
        if backend == "auto":
            backend = "gurobi" if HAVE_GUROBI else "highs"
        if backend == "gurobi":
            m = gp.Model(); m.Params.OutputFlag = 0
            m.Params.TimeLimit = sv["mip_time_limit"]; m.Params.MIPGap = sv["mip_gap"]
            x = m.addMVar(n, vtype=gp.GRB.BINARY); x.Start = x0
            m.addMConstr(A, x, "<", hi); m.addMConstr(A, x, ">", lo)
            m.setObjective(val @ x, gp.GRB.MAXIMIZE); m.optimize()
            sol, bound, gap = x.X, m.ObjBound, m.MIPGap
        else:
            res = milp(-val, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=LinearConstraint(A, lo, hi),
                       options={"time_limit": sv["mip_time_limit"], "mip_rel_gap": sv["mip_gap"], "disp": False})
            if res.x is None:
                return None, dict(mip_status=res.message, mip_time=time.time() - t0)
            sol, bound = res.x, -getattr(res, "mip_dual_bound", np.nan)
            gap = getattr(res, "mip_gap", np.nan)
        chosen = [(int(m_), int(s_)) for m_, s_, x_ in zip(vm_, vs_, sol) if x_ > 0.5]
        return chosen, dict(mip_vars=n, mip_rows=A.shape[0], mip_time=time.time() - t0, mip_bound=bound, mip_gap=gap, mip_backend=backend)

    def _exact_fixed(self, by_day, optional=False, start=None, time_limit=2.0, tol=CAP_TOL):
        """Exact solution when every listed game must be played in a known week (and, if `by_day`, on a known day and
        medium). The problem then separates into one small assignment MIP per week (or per week-day): place the group's
        games into its positions so that seat counts hold and no window exceeds its power cap. Valid when the rest rule
        is off (it is the only constraint linking weeks). `optional=True`: games may be left off television (each at most
        once) and seats are filled from the listed games. If the solver stops at its time limit without a solution, the
        group keeps its `start` assignment; if a group's power caps cannot be met at all, they are dropped for that group
        and the group is reported in `self.relaxed`."""
        inv = self.inv
        chosen, self.relaxed = [], []
        key_of_pos = inv.week[inv.pos_win] * 10 + (inv.day[inv.pos_win] if by_day else 0)
        for key in np.unique(key_of_pos[self.cs]):
            sel = np.nonzero(key_of_pos[self.cs] == key)[0]
            vm_, vs_ = self.cm[sel], self.cs[sel]
            games, gi = np.unique(vm_, return_inverse=True)
            pos, pj_ = np.unique(vs_, return_inverse=True)
            wins, wi = np.unique(inv.pos_win[pos], return_inverse=True)
            n = len(sel)
            seats = inv.pos_cap[pos]
            exact_fill = (len(games) >= seats.sum()) if optional else (len(games) == seats.sum())
            A = sparse.vstack([sparse.csr_matrix((np.ones(n), (gi, np.arange(n))), shape=(len(games), n)),
                               sparse.csr_matrix((np.ones(n), (pj_, np.arange(n))), shape=(len(pos), n)),
                               sparse.csr_matrix((self.P[vm_], (wi[pj_], np.arange(n))), shape=(len(wins), n))])
            caps = np.where(np.isfinite(self.win_cap[wins]), self.win_cap[wins] + tol, 1e9)
            g_lo = np.zeros(len(games)) if optional else np.ones(len(games))
            sol = None
            for attempt in (0, 1):
                lo = np.r_[g_lo, seats if exact_fill else np.zeros(len(pos)), np.zeros(len(wins))]
                hi = np.r_[np.ones(len(games)), seats, caps]
                res = milp(-self.V[vm_, vs_] - 1e-9, integrality=np.ones(n), bounds=Bounds(0, 1),
                           constraints=LinearConstraint(A, lo, hi),
                           options={"time_limit": time_limit, "mip_rel_gap": 1e-4, "disp": False})
                if res.x is not None:
                    sol = [(int(m), int(s)) for m, s, x in zip(vm_, vs_, res.x) if x > 0.5]
                    break
                if start is not None:                              # time limit without a solution: keep the incumbent
                    in_grp = set(zip(vm_.tolist(), vs_.tolist()))
                    sol = [(m, s) for m, s in start if (m, s) in in_grp]
                    if len(sol) == (seats.sum() if exact_fill else len(sol)):
                        break
                    sol = None
                caps = np.full(len(wins), 1e9)
                self.relaxed.append(int(key))
            if sol is None:
                raise RuntimeError(f"Fixed-matchup group {key} infeasible: {res.message}")
            chosen += sol
        return chosen

    # ---------------- evaluation and audit ----------------
    def evaluate(self, assignment):
        inv, vm = self.inv, self.vm
        if not assignment:
            return 0.0, 0.0, pd.DataFrame()
        m_ = np.array([m for m, _ in assignment]); s_ = np.array([s for _, s in assignment])
        lin = self.V[m_, s_]
        win = inv.pos_win[s_]
        G_act = np.bincount(win, minlength=len(inv.G)).astype(float)
        P_act = np.bincount(win, weights=self.P[m_], minlength=len(inv.G))
        L = vm.log_view(self.P[m_], self.D[m_], self.B[m_], self.R[m_], G_act[win], P_act[win], inv.wconst[win], inv.mconst[s_])
        real = np.exp(L)
        tab = pd.DataFrame({"week": inv.week[win], "day": inv.day[win], "time": [TIMES[t] for t in inv.pos_time[s_]],
                            "medium": [MEDIA[k] for k in inv.pos_med[s_]],
                            "team1": [self.teams[i] for i in self.pi[m_]], "team2": [self.teams[j] for j in self.pj[m_]],
                            "game_power": self.P[m_], "rivalry": self.R[m_].astype(bool),
                            "linear_view": lin, "realized_view": real, "pair": m_, "pos": s_}).sort_values(["week", "day", "time", "medium"])
        return float(lin.sum()), float(real.sum()), tab.reset_index(drop=True)

    def audit(self, assignment, required=None):
        """Independent re-check of every active constraint. Returns a list of violations (empty = feasible)."""
        inv, c = self.inv, self.c
        v = []
        st = self._empty_state()
        for m, s in assignment:
            if self.V[m, s] <= 0:
                v.append(f"pruned candidate used {m},{s}")
            ok = self._feasible(st, s, np.array([m]), lookahead=False)[0] and st["seat"][s] > 0
            if not ok:
                v.append(f"infeasible at {self.teams[self.pi[m]]}-{self.teams[self.pj[m]]} pos {s}")
            self._apply(st, m, s, +1)
        if c.get("min_exposure", 0) and (st["cnt"] < c["min_exposure"]).any():
            v.append(f"{int((st['cnt'] < c['min_exposure']).sum())} teams below min exposure")
        if required is not None and (st["used"] < required).any():
            v.append(f"{int((required & ~st['used']).sum())} required matchups not scheduled")
        return v

    # ---------------- driver ----------------
    def solve(self, rho, pairs=None, pair_week=None, pair_day=None, pair_medium=None, ref_power=None, seed=0, start=None,
              verbose=False, relin_iters=None):
        """Solve one season. When the cap is looser than σ = 1, the fixed-composition linearization over-rewards stacking
        strong games in one window, so (i) the σ = 1 schedule is kept as a candidate and (ii) the objective is
        re-linearized at the window composition the best schedule produced and re-solved (`relin_iters` times);
        the schedule with the highest realized viewership is returned."""
        it = self.relin_iters if relin_iters is None else relin_iters
        best = self._solve_once(rho, pairs, pair_week, pair_day, pair_medium, ref_power, None, seed, start, verbose)
        sig = self.c.get("power_cap_slack")
        if it and sig != 1.0 and self.c["matchup_mode"] in ("free", "subset"):
            # the σ = 1 schedule is feasible under a looser cap and its linearization is accurate: keep it as a candidate
            self.c["power_cap_slack"] = 1.0
            tight = self._solve_once(rho, pairs, pair_week, pair_day, pair_medium, ref_power, None, seed, None, verbose)
            self.c["power_cap_slack"] = sig
            if tight.realized > best.realized and not tight.stats["violations"]:
                best = tight
        pairs_used = None if (self.c["matchup_mode"] == "free" or pairs is None) else pairs
        for k in range(it):
            lp = best.table.groupby(self.inv.pos_win[best.table.pos.to_numpy().astype(int)]).game_power.sum()
            lin = self.lin_ref_default(ref_power).copy()
            lin[lp.index.to_numpy()] = lp.to_numpy()
            cand = self._solve_once(rho, pairs_used, pair_week, pair_day, pair_medium, ref_power, lin, seed + k + 1,
                                    list(zip(best.table.pair.astype(int), best.table.pos.astype(int))), verbose)
            if cand.realized > best.realized + 1e-9 and not cand.stats["violations"]:
                best = cand
            else:
                break
        return best

    def lin_ref_default(self, ref_power):
        return self.inv.Ptot.copy() if ref_power is None else np.asarray(ref_power, float)

    def _solve_once(self, rho, pairs, pair_week, pair_day, pair_medium, ref_power, lin_power, seed, start, verbose):
        t0 = time.time()
        c, sv = self.c, self.sv
        rho = np.asarray(rho, float)
        mode = c["matchup_mode"]
        if mode == "free" or pairs is None:
            pairs = np.column_stack(np.triu_indices(self.N, 1))
        pairs = np.asarray(pairs, int)
        self._prepare(rho, pairs, pair_week, pair_day, pair_medium, ref_power, lin_power)
        required = np.ones(self.M, bool) if mode in ("fixed_week", "fixed_season", "historical") else None
        rng = np.random.default_rng(seed)
        exact = mode == "historical" or (mode in ("fixed_week", "subset_week") and not c.get("rest_rule", True)
                                         and sv.get("exact_fixed", True) and not c.get("min_exposure", 0))
        if exact:
            self.tol = 1e-6 if mode == "historical" else CAP_TOL       # the 2019 reconstruction must match exactly
            best = self._exact_fixed(by_day=(mode == "historical"), optional=(mode == "subset_week"), start=start, tol=self.tol)
            lin, real, tab = self.evaluate(best)
            stats = dict(linear=lin, realized=real, games=len(best), seats=self.inv.n_seats, exact=True,
                         cap_relaxed_groups=list(self.relaxed),
                         violations=[] if self.relaxed else self.audit(best, required), time=time.time() - t0)
            self.tol = 1e-9
            return ScheduleResult(tab, lin, real, stats)
        if mode == "fixed_season" and start is not None:
            self.tol = CAP_TOL                          # the start may come from the exact MIP, which allows CAP_TOL
        best, best_val = None, -1.0
        for r in range(max(1, sv["grasp_starts"])):
            noise = np.exp(sv["rcl_noise"] * rng.standard_normal(len(self.cv))) if r else 1.0
            if r == 0 and start is not None:
                assign = list(start)
            else:
                assign = self._greedy(np.argsort(-self.cv * noise, kind="stable"))
            st = self._state_from(assign)
            self._reopt_seats(st, rng, sv["improve_passes"])
            if sv.get("swap_moves", True):
                if self._swap_moves(st, "all" if mode == "fixed_season" else "week"):
                    self._reopt_seats(st, rng, 1)
            if c.get("min_exposure", 0):
                self._repair_exposure(st, c["min_exposure"])
            assign = [(m, s) for s in range(self.inv.S) for m in st["assign"][s]]
            val = self.V[[m for m, _ in assign], [s for _, s in assign]].sum() if assign else 0.0
            if val > best_val:
                best, best_val = assign, val
        stats = dict(heuristic_linear=best_val, heuristic_time=time.time() - t0, candidates=len(self.cv), matchups=self.M)
        use_mip = sv["use_mip"]
        if use_mip or (required is not None and len(self.audit(best, required))):
            chosen, info = self._mip(best, required)
            stats.update(info)
            if chosen is not None and self.V[[m for m, _ in chosen], [s for _, s in chosen]].sum() >= best_val - 1e-6 \
                    and not self.audit(chosen, required):
                best = chosen
        lin, real, tab = self.evaluate(best)
        stats.update(linear=lin, realized=real, games=len(best), seats=self.inv.n_seats, violations=self.audit(best, required),
                     time=time.time() - t0)
        self.tol = 1e-9
        if verbose:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items() if k != "violations"},
                  "violations:", stats["violations"][:3])
        return ScheduleResult(tab, lin, real, stats)

def matchup_appeal(sch):
    """Matchup-only part of predicted log viewership at a typical window (5 games, 550 total power, Saturday afternoon)."""
    return VM.log_view(sch.P, sch.D, sch.B, sch.R, 5.0, 550.0, 0.0, 0.0)


def pattern_tv(sch, rho, pairs, slot_order, pair_week=None, ref_power=None):
    """2019-style TV rule. `slot_order`: list of (q, position) for the 2019 televised games, most appealing first, where q
    is the game's appeal rank among all 2019 FBS games as a fraction (0 = most appealing game of the season). Each 2019
    televised game is replaced by this season's game at the same appeal quantile (the nearest free one if that game's
    teams are already televised that week) and put in the same position. Returns a ScheduleResult (realized)."""
    inv = sch.inv
    rho = np.asarray(rho, float)
    pairs = np.asarray(pairs, int)
    saved = sch.c["matchup_mode"]
    sch.c["matchup_mode"] = "subset"                     # build P, D, B, R without pinning or pruning side effects
    sch._prepare(rho, pairs, None, ref_power=ref_power)
    sch.c["matchup_mode"] = saved
    order = np.argsort(-matchup_appeal(sch), kind="stable")        # order[r] = game with appeal rank r
    M = len(order)
    pi, pj = sch.pi, sch.pj
    busy = np.zeros((sch.N, inv.nW), bool)
    used = np.zeros(M, bool)
    pw = None if pair_week is None else np.searchsorted(inv.weeks, np.asarray(pair_week))
    assign = []
    for q, s in slot_order:
        w = inv.pos_week[s]
        r0 = min(int(q * M), M - 1)
        for d in range(M):
            hit = None
            for r in ((r0 + d, r0 - d) if d else (r0,)):
                if 0 <= r < M:
                    m = order[r]
                    if not used[m] and not busy[pi[m], w] and not busy[pj[m], w] and (pw is None or pw[m] == w):
                        hit = m
                        break
            if hit is not None:
                used[hit] = True; busy[pi[hit], w] = busy[pj[hit], w] = True
                assign.append((int(hit), int(s)))
                break
    lin, real, tab = sch.evaluate(assign)
    return ScheduleResult(tab, lin, real, dict(games=len(assign), rule="2019-style"))
