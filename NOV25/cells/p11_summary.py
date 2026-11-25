# %% [md]
# ## 12. Summary of findings
# The numbers referenced in the paper's Results section.

# %%
A = W.A_rank.mean(); C = W["style2019:sq"].mean(); D = W["optimized:sq"].mean(); F = W.F_max.mean()
print(f"Actual 2019 slate (2019 teams, conferences and ratings):         {HIST.realized:7.1f}M")
print(f"  re-slotted within weeks (Model 0):                             {SCEN_RESULTS['Model 0 (2019 games, reslot within week)'].realized:7.1f}M "
      f"({100 * (SCEN_RESULTS['Model 0 (2019 games, reslot within week)'].realized / HIST.realized - 1):+.1f}%)")
print(f"  2019 schedule, optimal choice of televised games (Model 0b):   {SCEN_RESULTS['Model 0b (2019 slate, choose TV games within week)'].realized:7.1f}M "
      f"({100 * (SCEN_RESULTS['Model 0b (2019 slate, choose TV games within week)'].realized / HIST.realized - 1):+.1f}%)")
print(f"  free matchups (Model 4-sigma):                                 {SCEN_RESULTS['Model 4-sigma (no conference structure)'].realized:7.1f}M "
      f"({100 * (SCEN_RESULTS['Model 4-sigma (no conference structure)'].realized / HIST.realized - 1):+.1f}%)")
print(f"\nSimulation, mean per season ({RUN['seasons']} seasons × {RUN['reps']} replications):")
print(f"  A. 2019 televised games by rank, as slotted:                   {A:7.1f}M   (= 100)")
print(f"  C. status quo, 2019-style TV:                                  {C:7.1f}M   {100 * C / A:6.1f}")
print(f"  D. status quo, optimized TV:                                   {D:7.1f}M   {100 * D / A:6.1f}")
for k in POL:
    e0, e = W[f"style2019:{k}"].mean(), W[f"optimized:{k}"].mean()
    print(f"  {NAMES[k]:<22} 2019-style TV {e0:7.1f}M ({100 * (e0 / C - 1):+5.1f}% vs C) | optimized TV {e:7.1f}M "
          f"({100 * (e / D - 1):+5.1f}% vs D, {100 * (e / A - 1):+5.1f}% vs A)")
print(f"  F. theoretical maximum:                                        {F:7.1f}M   {100 * F / A:6.1f}")
print("\nPath decomposition (best policy):")
print(PATH.round(1).to_string(index=False))
print(f"\nOutputs in {OUT_DIR.resolve()}: " + ", ".join(sorted(p.name for p in OUT_DIR.iterdir())))

# %% [md]
# ## 13. Notes for the paper (where the code and the current draft differ)
# 1. **Baseline.** Report policies against two baselines: the 2019 televised games mapped by preseason rank and slotted
#    as in 2019 (the status quo as practiced), and the status quo with optimized television. The prototype reported only
#    the second, which already contains most of the available gain.
# 2. **Decomposition.** The 2 × 2 table (Section 9.1) separates *optimization* (which games are televised and where) from
#    *policy* (which games exist). Most of the total gain over the 2019 slate is optimization; the policy adds a further,
#    smaller increment, and the increment is larger with optimized television than with 2019-style choices.
# 3. **Window cap (A.8).** In the simulation the cap is defined per season from the 2019 games at the same ranks. It is
#    active in every window, but loosening it (σ = 1.10, or no cap) does not raise realized viewership (Section 10): the
#    model's own competition terms (games on, share of window power) already penalize stacking strong games together.
#    The cap is therefore not what limits the policy gains.
# 4. **Policies (Section 7).** Movement uses a fixed number of promoted/relegated teams by standings (or a full re-sort, as
#    a tuned alternative). The tuned settings and their plain-language rules are in Section 11 / `policy_rulebook.md`.
# 5. **Dynamics (Section 5, Appendix C).** Era levels are drawn around each program's long-run level (λ calibrated);
#    against the supplied 1960–2019 Massey panel this matches the distribution of rating variances far better than era
#    levels drawn around zero. The supplied file has no 1991 season and 81 (not 113) programs with complete records.
# 6. **Rest rule.** The 2019 televised slate violates the rest rule once, so the rungs that keep the 2019 games (A, B, D′)
#    run without it; every optimized simulated season uses it.
# 7. **2019 kickoff windows** are reconstructed (the game file has dates and media, not kickoff times); the reconstruction
#    matches the inventory exactly.
