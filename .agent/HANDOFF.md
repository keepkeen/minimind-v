# BudgetLab V2 handoff

2026-10-09: implemented a new dense offline budget-distillation entry, dynamic padding,
global answer-token normalization across accumulation/DDP, shared frozen raw visual
features, config roundtrip and paired image-cluster evaluation. Details and executable
commands: `docs/BUDGET_DISTILLATION_V2.md`; evidence: `docs/VALIDATION_V2.md`.

45 maintained CPU tests passed in the first complete run. Real pinned upstream weights
loaded, six generation prefixes collected and one real 64→16 optimizer update completed;
native checkpoint reload passed. No held-out accuracy or GPU speedup conclusion.

V1 training scripts retain legacy accumulation semantics. Do not call the new offline
objective LT-OPD/SCOPD. Leave user/workspace numeric-suffix source copies untouched and
unstaged. No downloaded weights or smoke-only weights belong in Git.

Next useful research: independent image/scene splits and matched CE64/CE16/KD16 seeds,
then diagnose missing visual information versus unreliable use before adding mechanisms.
