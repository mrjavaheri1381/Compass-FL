# COMPASS-FL codebase audit (Phase 1)

> **Revision 2.** Sections 1–5 are the original audit. The decisions on Q1–Q14 are in "Decisions taken" (after section 5). Revision-2 items, including new gap rows, the environment data, new risks, new paper issues and open questions RQ1–RQ6, are in **section 6**. Where they conflict, section 6 and the decisions table take precedence over sections 2–5.

Scope: `dlsim/compass/`, `simulations/compass/`, and the Plexus/core code they depend on (`dlsim/core/`,
`simulations/learning_simulation.py`, `simulations/bandwidth_scheduler.py`, `simulations/args.py`, `pyipv8/simulation/`),
compared against `main.tex` (the specification) and `CLAUDE_CODE_TASK.md` (the checklist).

What I ran to ground this audit:

- A smoke run (`simulations/compass/cifar10.py`, N=20, n=4, M=10, `--bypass-training --bypass-model-transfers --he-mode plain`,
  5 rounds). It completes in ~36 s of simulated time and writes `selections.csv`, `proposals.csv`, `rounds.csv`.
- The same run twice with the same seed: `selections.csv` (decision columns) and `derived_samples.csv` were identical.
  The current code is deterministic in practice, but the seed does not control everything (see §3, R-15 to R-17).
- TenSEAL: `CKKSVector.dot()` hides the non-first slots at the API level (`decrypt()` returns 1 value), so the paper's
  mask cannot be demonstrated or tested through the high-level API. The low-level `tenseal.sealapi`
  (`Encryptor.encrypt_zero`, `Evaluator.rotate_vector`, `multiply_plain`) is available and makes a paper-exact
  implementation possible.
- There are no tests outside `pyipv8/`, and `pytest` is not installed in the active Python 3.9.6 environment.

---

## 1. Map of the codebase

### 1.1 Modules

| Module | Main classes / functions | Role |
|---|---|---|
| `dlsim/compass/community.py` (1283 lines) | `CompassCommunity`, `AggregationState` | The per-node protocol. It covers the round state machine, typed EVA messages, the HE exchange, client and aggregator roles, the population-view update, selection and handover, and bootstrap. |
| `dlsim/compass/selection.py` | `SelectionOrchestrator`, `compute_proposal`, `Proposal`, `SelectionOutcome` | Aggregator-side selection (collect → filter → propose → rank → hash fallback) and the candidate-side local solve. |
| `dlsim/compass/fitness.py` | `SelectionContext`, `CobbDouglasEvaluator`, `validate_weights` | The utility: five Φ components and a Cobb–Douglas product in log space. |
| `dlsim/compass/spma.py` | `SPMASolver`, `Chromosome` | A memetic GA with a fixed-cardinality encoding. It is the only solver. |
| `dlsim/compass/he.py` | `HESimilarityManager` | CKKS (TenSEAL) similarity in two summation modes, `masked` (default) and `rotation`, plus a `plain` mode. |
| `dlsim/compass/peer_manager.py` | `CompassPeerManager` | The population view: Plexus membership plus f, B, (round, Q), τ and a round-time EMA, with `get_view` and `merge_view`. |
| `dlsim/compass/sample_manager.py` | `SampleManager` | The hash pool `TopM md5(pk-t)`, a next-M-in-order resample, and the per-(pk, round) solver seed. |
| `dlsim/compass/caches.py` | `PingRequestCache`, `SelectionPhaseCache` | Ping RTT capture, and the request/response phase with timeout. |
| `dlsim/compass/payloads.py` | `AdvertiseMembership`, `Ping`, `Pong`, `AggAck` | IPv8 payloads. |
| `simulations/compass/compass_simulation.py` | `CompassSimulation` | Settings from argparse, static capabilities from FedScale traces, injection of ground-truth latencies, HE bootstrap, accuracy logging, CSV flushing. |
| `simulations/compass/bypass_network_community.py` | `CompassBypassNetworkCommunity` | Simulated transfers: a sleep for the one-way latency, then a BW-scheduled transfer. |
| `simulations/compass/{cifar10,femnist,celeba}.py` | — | Entry points (argparse). |
| `dlsim/core/*` (shared with Plexus) | `ModelTrainer`, `ModelManager`, `FedAvg`, datasets, models, `SessionSettings` + `CompassSettings` | Training, which also advances simulated time, aggregation, data, and settings. |
| `simulations/learning_simulation.py` | `LearningSimulation` | The discrete event loop (`DiscreteLoop`), IPv8 nodes, traces, the latency matrix, and statistics. |

### 1.2 Data flow of one round (as implemented)

```
A(t): aggregator_complete_round(t)            community.py:1065
  ├─ FedAvg (UNWEIGHTED) over received models  :1080
  ├─ update_population_after_round             :1095   τ for contributors, Q with an adaptive target
  └─ run_selection_and_handover(t)             :1155
       └─ SelectionOrchestrator.run()          selection.py:197
            ├─ pool = TopM md5(pk-(t+1)) over A(t)'s active view
            ├─ collect: sim_request → candidates reply {row, f, b, q, tau}   (self-reported)
            ├─ view ← self-reported f, b, q, tau                              selection.py:224-231
            ├─ S matrix = average of S_ij and S_ji                            :343
            ├─ eligible = f ≥ f_min ∧ b ≥ b_min
            ├─ sel_request (pool, S, f, q, tau, f_max, n) + full view → each eligible candidate
            │     candidate: N_a = {L ≤ L_max}, SPMA on Cobb–Douglas → Proposal(a, C, U)
            ├─ rank: (|C| == n, U) desc;  no proposal → next M in hash order; still none → hash fallback
       └─ handover(): model_handover(θ(t), view, clients, backups) to ranked[0], then ranked[1], ...
A(t+1): start_round_as_aggregator(t+1) → aggregated_model(θ(t), view) → clients (+ itself: aggregator_trains)
C(t+1): train (sleep 3·local_steps·ms_per_sample) → trained_model(+view) → A(t+1) or a client-side backup aggregator
A(t+1): completes at success_fraction·expected models or at aggregation_timeout (300 s)
```

### 1.3 Where each paper concept lives

| Paper concept | Location | Note |
|---|---|---|
| Hash pool M^(t) | `sample_manager.py:27` | done |
| Bootstrap A^(0) | `compass_simulation.py:279`, `community.py:1268` | uses the round-0 hash order, not M^(1) |
| Population view | `peer_manager.py` | missing m_i and κ_i; written by everyone |
| ŷ_i, S_ij | `he.py:31-43` | done |
| r_i, Φ_repr, Φ_coh, w_c, w_r | — | **missing** |
| Q EMA | `community.py:1108-1137` | adaptive target, hard indicator, Q(0)=0.5 |
| f, B estimation (t_ack/t_first/t_last, β_i, m_i) | — | **missing** (f and B are static values from traces) |
| Utility | `fitness.py` | 5 components, but Φ_data, Φ_fair and Φ_comp differ from the paper |
| Neighborhood N_a (k_nb, ties, decline) | `selection.py:108-122` | threshold only |
| Latency probing (P probes, median RTT) | `community.py:224`, `caches.py` | single ping, RTT/2, EMA; by default **oracle-injected** |
| Solver (Algorithm 1) | — | **missing** (SPMA instead) |
| Verification | — | **missing** |
| Fault handling | `community.py:1205` (handover retry), client-side backups | a different mechanism from the paper |
| Deadline δ_dl, grace | — | **missing** (uses success_fraction + timeout) |
| FedAvg weighted by D_i | `community.py:1080` → `fedavg.py:14` | **unweighted** |
| CKKS protocol (rotations + e_1 mask + flooding) | `he.py` | the default is a zero-sum additive mask; no e_1, no flooding |
| Solver / selector interfaces | — | **missing** |
| Device classes, latency model, failures | — | **missing** (FedScale traces + `latencies.txt`) |
| Config system | `simulations/args.py`, `CompassSettings` | argparse only; no YAML, presets or `run.py` |
| Tests | — | **none** |

---

## 2. Gap table

Status: `done` / `partial` / `missing` / `wrong`. File:line points at the current code.

### A. System model and node state

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| A1 | Graph of N nodes, no fixed server, roles aggregator/client/idle | partial | `community.py:936-969` | The aggregator also trains by default (`aggregator_trains=True`). The paper has C ⊂ M\{a}, so the aggregator is not a client. Default to off and keep the flag (Q1). |
| A2 | D_i, y_i, ŷ_i = y_i/‖y_i‖₂ | done | `he.py:31`, `datasets/*.label_counts` | Expose D_i to the protocol, which needs it for FedAvg and f̂. |
| A3 | S_ij = ŷ_iᵀŷ_j ∈ [0,1], S_ii = 1 | done | `he.py:41`, `selection.py:352` | — |
| A4 | r_i = Σ_{j∈V} S_ij (incl. j=i), updated on join | missing | — | Compute it from the node's full row plus S_ii = 1, and update it on join (and on leave, Q6). `he_max_peers` (`community.py:279`) truncates the row, which makes r_i wrong. Forbid it or warn. |
| A5 | Population view {τ, Q, f, B, m, κ, membership}, travels A→A, written only by aggregators | wrong | `peer_manager.py:77-112`, `selection.py:224-231`, `community.py:476, 878, 978` | Several fixes. (a) Add m_i and κ_i. (b) Make the view authoritative only through the handover: no `merge_view` from clients' `trained_model` messages, membership adverts, or `sel_request`. (c) Stop overwriting the view with self-reported f, b, q and tau from `sim_response`. (d) Stop sending the view to clients (`aggregated_model`) and to every candidate (`sel_request`). Membership can stay Plexus-style, but the selection state must not. |
| A6 | Δτ_i = t − τ_i, τ_i = 0 if never | done | `peer_manager.py:65`, `selection.py:138` | — |
| A7 | Q EMA, participants only, Q(0) = 1, contribution exp(−γ·max(0, δ−δ_dl)), δ=∞→0 | wrong | `community.py:1115-1137`, `session_settings.py:87` | Several fixes. Use Q(0)=1 (currently 0.5). Use a fixed δ_dl from config instead of `round_target_factor × EMA(median)`. Make the soft exponential the formula (the default currently multiplies by an indicator, so any late update gives 0). Measure δ_i from dispatch. Count late arrivals within the grace period. Drop the round-time EMA. |
| A8 | f, B estimation: class priors, t_ack/t_first/t_last, f̂ = E·D·c/T_cmp, B̂ = Z/T_up, β_i = max(β, 1/(m+1)), m += 1, no measurement past grace | missing | `compass_simulation.py:191-229`, `community.py:476` | Currently f and B are static ground truth from FedScale traces, injected into every view and also self-declared in membership adverts. Implement class priors, timestamp capture in the simulator (consistent with ground truth), the estimator, and the update rule. |

### B. Utility

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| B1 | Φ_comp = min(1, min f / f_ref) | wrong | `fitness.py:96-100`, `selection.py:305` | It normalizes by f_max, the maximum f in the aggregator's view and pool, which varies by round and view. Use the config constant f_ref and min(1, ·). |
| B2 | Φ_comm = 1 − max L_ia / L_max | partial | `fitness.py:102-106` | The formula is right, but N_a uses `≤ L_max` (`selection.py:115`), so Φ_comm can be 0. L is one-way (RTT/2), while the paper uses RTT (Q4). |
| B3 | Φ_coh = mean_{i<j∈C} S_ij | wrong | `fitness.py:108-119` | It is implemented as `Φ_dist` over C ∪ {A}, which includes the aggregator. It must cover C only. |
| B4 | Φ_repr from S, r and the network-wide estimate (N/M)·Σ_pool r_j | missing | — | Implement it. The current aggregator computes the estimate once per round and ships it in the context. |
| B5 | Φ_data = Φ_coh^{w_c} · Φ_repr^{w_r}, w_c = 0 allowed | missing | — | Add w_c and w_r to config and validate w_c, w_r ≥ 0 with w_c + w_r > 0. |
| B6 | Φ_fair = mean min(Δτ, Δτ_cap)/Δτ_cap | wrong | `fitness.py:72-73, 121-125` | It normalizes by the max staleness in the candidate's own neighborhood, so the score depends on the candidate and utilities are not comparable across proposals. Use the config Δτ_cap. |
| B7 | Φ_strag = mean Q | done | `fitness.py:128` | — |
| B8 | Clip every component to [ε, 1] | partial | `fitness.py:19, 149` | Clipping happens only inside the product. The returned and logged components are unclipped, and ε is hard-coded. Clip in the component function and make ε configurable. |
| B9 | Multiplicative (default) and additive forms, w_m > 0, Σ = 1, fail loudly | partial | `fitness.py:22-33, 138-151`, `session_settings.py:100-103` | Only the multiplicative form exists. The validation accepts w_m = 0, and the paper requires w_m > 0. Add the additive form and require strict positivity. Validation is duplicated; keep one copy. |
| B10 | f_ref, L_max, Δτ_cap network-wide config constants | partial | — | L_max exists (`max_latency`). Add f_ref and Δτ_cap. |
| B11 | One pure function returns U and all components, shared by candidates, verifier and all solvers | partial | `fitness.py:87-154` | `SelectionContext` is close, but it is tied to the SPMA evaluator, and there is no verifier. Refactor it into a pure `evaluate(problem, C, a) → (U, components)` plus an incremental-state API for `greedy_swap`. |
| B12 | Partial sets use \|C\| in place of n | partial | `fitness.py:94` | `components()` already uses len(selected). Define Φ_coh for \|C\| = 1 explicitly (Paper issue P-13). |

### C. Round protocol

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| C1 | Hash pool TopM H(i‖t+1); option to use an unpredictable seed (hash of θ) | partial | `sample_manager.py:18-32` | Add the seed mode `round` / `model_hash`. |
| C2 | Bootstrap: the smallest-hash member of M^(1) is A^(0), which starts at context collection | partial | `compass_simulation.py:299-305` | It currently uses the first eligible node in the hash order of round 0. Use argmin H(i‖1) over M^(1). |
| C3 | Context collection: only r_i and the row restricted to the pool; f, B, Q, Δτ from the view; S_ij vs S_ji tolerance check that flags violators | wrong | `community.py:337-348`, `selection.py:224-231, 343-361` | Remove f, b, q and tau from the reply, add r_i, replace averaging with the check (flag, then policy: drop the pair or the node, Q9), and log flags. |
| C4 | Eligibility f ≥ f_min ∧ B ≥ B_min from the view | partial | `selection.py:234-237`, `compass_simulation.py:209-212` | It uses self-reported f and b. Read them from the view (estimates). Thresholds may stay percentile-derived as a config option, but absolute values must be possible from config. |
| C5 | Probing: P probes to every pool member, median RTT, cost in the simulated clock, optional overlap with the previous round | wrong | `community.py:215-227, 749-751`, `compass_simulation.py:231-248` | By default every node holds **ground-truth latencies** (`--inject-latencies` defaults to True), which is oracle knowledge. Otherwise there is a single ping with an EMA of RTT/2. Implement P probes, the median, noisy samples from the latency model, message and byte accounting, simulated time, and a prefetch option. |
| C6 | N_a = k_nb nearest with L < L_max, ties by id, decline if \|N_a\| < n | wrong | `selection.py:108-122` | There is no k_nb and no tie rule, and the code shrinks n to \|N_a\| and proposes smaller groups instead of declining. |
| C7 | Local solve with the configured solver | wrong | `selection.py:152-162` | It is hard-wired to SPMA. Use the solver registry. |
| C8 | Proposal (a, C_a, U_a, {L_ia}) | partial | `selection.py:43-58` | Add the latencies. |
| C9 | Verification: recompute U from the context and the proposal's latencies, discard on mismatch | missing | — | Implement it and log rejected proposals. |
| C10 | Select the max verified U and keep the ranked list | partial | `selection.py:253` | It currently prefers full-size groups, then U. Once groups always have \|C\| = n, rank by U, tie-broken by id. |
| C11 | Handover θ(t) and the view to A^(t+1) | done | `community.py:1205-1236` | — |
| C12 | Fault handling: keep θ, the view and the ranked list until A^(t+1) acks and completes; on timeout or failure, fall back to the next-ranked; with no proposals, resample with the next hash index; last resort, serve another round itself | partial | `community.py:1155-1236`, `selection.py:203-265, 363-375` | A(t) drops its state as soon as the handover message is delivered. Failure during round t+1 is handled by client-side backup aggregators, a mechanism that is not in the paper. Resampling takes the next M in the same order, and the last resort is a Plexus hash fallback rather than self-serve. Implement the paper mechanism: a completion notice from A^(t+1) to A^(t), a watchdog, the next-ranked proposal, the next hash index, and self-serve. Keep client backups only as an optional flag (Q12). |
| C13 | Local training E epochs; the client returns θ_i and D_i | wrong | `model_trainer.py:50-53`, `community.py:876-878` | It uses `local_steps` mini-batches, and D_i is not sent. |
| C14 | Deadline δ_dl from dispatch; late updates excluded but timed within the grace period | missing | `community.py:998-1001, 1008-1014, 1047-1063` | It completes on `success_fraction` or the 300 s timeout. Late models after completion are released without recording arrival. |
| C15 | FedAvg weighted by D_i over on-time updates; keep θ if none | wrong | `community.py:1080`, `fedavg.py:14-15` | Weights are uniform. The empty-set case is done (`:1085`). |
| C16 | State updates: τ ← t for **every selected client**, then Q, f and B per A | wrong | `community.py:1102-1103` | τ is set only for contributors and for the aggregator itself, so a selected client that is missing keeps a high Δτ and is favored again. This contradicts the paper. |
| C17 | Round indexing (A^(t) trains C^(t) from θ^(t−1), selects t+1) | done | `community.py:1159`, `:1268-1279` | Round 0 is a completed pseudo-round. That is fine. |

### D. Solvers

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| D0 | `Solver` protocol, `SelectionProblem`, `Budget`, `SolverResult`, registry by name | missing | — | Create `dlsim/compass/solvers/` with a registry, so that a new solver needs only a new file plus a decorator. |
| D1 | `greedy_swap` (Algorithm 1, incremental O(1) swaps, deterministic) | missing | — | Implement it. |
| D2 | `exact` (B&B over bottleneck thresholds, refuses above a size limit) | missing | — | Implement it. |
| D3 | `tabu` | missing | — | Implement it. |
| D4 | `simulated_annealing` | missing | — | Implement it. |
| D5 | `genetic` (fixed-size set encoding + repair) | partial | `spma.py` | SPMA is a fixed-cardinality memetic GA. Wrap it as `genetic` and add the budget and the new interface. |
| D6 | `lns` | missing | — | Implement it. |
| D7 | `random` | missing | — | Implement it. |

### E. Selectors

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| E0 | `Selector` interface | missing | — | Create `dlsim/compass/selectors/`. |
| E1 | `compass` | partial | `selection.py` | Refactor it behind the interface. |
| E2 | `plexus_random` | partial | `selection.py:363-375`; the original in `dlsim/plexus/` | The hash fallback is essentially Plexus selection. Expose it as a selector inside the same protocol so the clock and logging are comparable. Keep the original Plexus simulation untouched. |
| E3 | `power_of_choice` | missing | — | It needs local losses (Q8). |
| E4 | `oort_like` | missing | — | Statistical utility × system speed (Q8). |
| E5 | `oracle_central` | missing | — | A full-network, full-knowledge solve (Q8). |
| E6 | Ablations purely by config | partial | `args.py:61-100` | The weights exist. w_c, w_r, the form, the solver, k_nb and f_ref do not. |

### F. Privacy-preserving similarity

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| F1 | Modes `plaintext` / `ckks` | partial | `he.py:25-26, 124, 145-147` | `plain` still runs the whole message exchange with the vectors in the clear. That is fine as an overhead baseline, but the spec's `plaintext` means computing S directly (no protocol). Provide `plaintext` (direct, default for FL sims), and keep `plain_protocol` optional. |
| F2 | CKKS exactly as in the paper: K slots, slot-wise × ŷ_j, ⌈log₂K⌉ rotations, × e_1, + flooding Enc(0), return | wrong | `he.py:111-120, 150-160, 176-179, 188-191` | The default `masked` mode adds a zero-sum random mask and the initiator decrypts all K slots; there are no rotations, no e_1 mask and no flooding. `rotation` uses TenSEAL `dot`, also with no e_1 and no flooding. The public context omits the public key (`save_public_key=False`), so the responder cannot produce the fresh Enc(0). Re-implement it with `tenseal.sealapi`. |
| F3 | Join-time execution in both directions; answer each identity once | partial | `community.py:271-335` | Reciprocation exists (`he_reciprocate`). "Answer once per identity" is not enforced (`on_he_vector` answers every request), and the newcomer's ciphertext is not stored publicly. |
| F4 | Optional DP noise on the returned similarity | missing | — | Add config: on/off and scale. |
| F5 | Benchmark script (enc/eval/dec time, message sizes, |error| vs K) | missing | — | Add `scripts/bench_ckks.py`. |
| F6 | Unit test: without the mask, the extra slots leak; with it they are ~0 | missing | — | Needs the low-level API (see above). |

### G. Simulation environment

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| G1 | Simulated clock; round time from modeled compute, transfer and latency | partial | `DiscreteLoop`, `model_trainer.py:62-81` | The clock is fine. Compute time = `3.0 × local_steps × ms_per_sample`, which ignores D_i, E, the batch size and c, and conflicts with f = E·D·c/T. Replace it with T_cmp = E·D_i·c/f_i (true f, optional jitter). |
| G2 | Device classes (distributions of f and B, failure probability, priors f̄ and B̄) | missing | `learning_simulation.py:136-157` | Add class-based generation from config. FedScale stays an optional source. Separate the uplink and downlink limits (the BW scheduler has one `bw_limit`). |
| G3 | Latency model (clustered regions + noise); noisy probes | partial | `learning_simulation.py:206-228` | It is a site matrix from a file, round-robin, truncated to whole ms (`int(...)`, `:225`), with no noise. Add a `clustered` model and keep `file`. |
| G4 | Churn: join (triggers HE), leave/fail, aggregator failure | partial | `core/community.py:63-85`, `community.py:352-411` | Joining and leaving come only from an availability-trace pickle. Add config-driven failure processes, including targeted aggregator failure. |
| G5 | Data: CIFAR-10/100 Dirichlet(α); FEMNIST/Shakespeare hooks; small CNN / ResNet configurable | partial | `datasets/__init__.py`, `models/__init__.py:19-31`, `Partitioner.py:64,144,191` | CIFAR-100 and Shakespeare are missing. For cifar10, `create_model` returns **None** whenever `--model` is set. Partitioner seeds are hard-coded (1234/1235) and not tied to the run seed. Add a small ResNet. |

### H. Logging and outputs

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| H1 | Per-round structured record (round, sim time, acc/loss at an interval, aggregator, clients, all Φ incl. coh/repr, U, #candidates/eligible/proposals/rejected/fallbacks, solver evals/time, on-time/late/missing, context and probe msgs/bytes, f/B estimate error) | partial | `community.py:1238-1266, 1145-1153`, `compass_simulation.py:24-30` | The data is spread over 4 CSVs, keyed by the logging peer, with no client IDs, rejected count, late count, per-round bytes or estimate error. Add one `rounds.jsonl` per run. |
| H2 | Save the resolved config, seed and git commit | partial | `session_settings.py:157-164` | `settings.json` is written, but not the args or the git hash. |
| H3 | Plot script (accuracy vs sim time, Jain fairness, ...) | missing | — | Add `scripts/plot_compass.py`. |

### I. Configuration and CLI

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| I1 | One config system with every paper parameter | partial | `args.py`, `session_settings.py:38-107` | Missing parameters: n separate from the Plexus `sample_size`, k_nb, f_ref, Δτ_cap, β, ε, δ_dl, grace, E, w_c, w_r, R_s, I_max, P, utility_form, solver, selector, DP, device classes, latency model. |
| I2 | CLI overrides like `solver=tabu weights.comm=0.3 seed=3` | missing | — | Use YAML plus dotted `key=value` overrides (PyYAML is installed; no new dependency). |
| I3 | `configs/` with `smoke.yaml`, `default.yaml` and one per baseline; `run.py --config` | missing | — | Create them. |

---

## 3. Bugs and risks not covered by the checklist

| ID | Issue | Where | Impact |
|---|---|---|---|
| R-1 | Candidates self-report f, b, Q and τ, and the aggregator writes these into the view. A node can inflate its own reliability and staleness. | `selection.py:224-231`, `community.py:337-348` | Breaks the paper's integrity claim (Sec. 4.7). |
| R-2 | The view is merged from **every** received message: clients' trained models, aggregated models, selection requests. `update_reliability` keeps the entry with the highest round stamp, so any node can overwrite any entry. | `community.py:685-686`, `peer_manager.py:90-112` | Same as R-1. It also leaks every node's Q, f and B to every client. |
| R-3 | The full population view (O(N)) is sent with every `sel_request` and every `aggregated_model`. | `selection.py:325-326`, `community.py:978` | Inflates selection and model bytes, which skews the overhead results. |
| R-4 | Proposals with \|C\| < n are accepted (n is shrunk to \|N_a\|). | `selection.py:119` | The group size is not guaranteed. |
| R-5 | Per-candidate normalization (`f_max` from the view, `staleness_max` per neighborhood) makes utilities **incomparable across candidates**, so the argmax over proposals is not meaningful. | `fitness.py:72`, `selection.py:305` | Selection quality. |
| R-6 | Missing similarities default to 0.5 silently. | `fitness.py:60, 78-81` | Hides HE failures. Log them and make it a config value. |
| R-7 | The aggregator trains and is counted in Φ_dist. With `success_fraction=1` the expected count includes itself. | `community.py:966-969, 992-996` | Deviates from the paper's model. |
| R-8 | A late model after completion is ACKed but its arrival is not recorded, so Q treats it as δ = ∞ (score 0). The default hard indicator also gives 0 for any lateness. | `community.py:1008-1014, 1131-1134` | Q is biased low. |
| R-9 | `apply_fedscale_traces` takes compute from `data[device_ids[ind]]` but bandwidth from `data[ind + 1]`, two different devices (inherited from Plexus). | `learning_simulation.py:145, 152` | Unrealistic f–B pairing. Irrelevant once device classes replace it, but it should be fixed if FedScale remains an option. |
| R-10 | Latencies are truncated to whole milliseconds (`int(latency_ms)`), and their units are one-way, while the paper uses RTT. | `learning_simulation.py:225`, `caches.py:55` | Precision and semantics. |
| R-11 | `create_model("cifar10", architecture=X)` returns `None`. | `models/__init__.py:20-23` | Crash or silent failure when `--model` is set. |
| R-12 | The HE `masked` mode reveals ŷ_i ⊙ ŷ_j + r with a bounded uniform zero-sum mask. This is not statistically hiding: K−1 linear constraints are known, and with bounded masks individual products are partially recoverable. | `he.py:188-191` | Privacy claim. Superseded by the paper protocol. |
| R-13 | `he_max_peers > 0` gives incomplete similarity rows, so r_i and Φ_repr are wrong. | `community.py:279-281` | Correctness when that option is used. |
| R-14 | The training time model `3.0 × local_steps × ms/sample` ignores the batch size, so the "speed" units are inconsistent. | `model_trainer.py:15, 69` | The timing model cannot produce consistent f estimates. |
| R-15 | `model_seed` is never set from `--seed` (always 0), so the initial θ is the same for all seeds. | `session_settings.py:147`, `compass_simulation.py:126-143` | Seed sweeps are not independent. |
| R-16 | Partitioner seeds are fixed (1234/1235), so the data split is identical for every seed. The synthetic label counts use `model_seed`. | `Partitioner.py:64,144,191`, `community.py:249` | Same as R-15. |
| R-17 | Iteration over `set`s of `bytes` depends on `PYTHONHASHSEED` (e.g. `peers_to_inform`, `he_pending`). In the run above it did not affect decisions, but send order can affect BW-scheduler order and hence timing. | `community.py:379-393`, `caches.py:83` | Possible nondeterminism under contention. Sort, or set `PYTHONHASHSEED` in `run.py`. |
| R-18 | `solve_time` is wall-clock and the solver's compute cost is not modeled in the simulated clock. Fine as a metric, but it must never feed the clock. | `selection.py:104` | Determinism. Keep it log-only. |
| R-19 | The A(t) → A(t) self-handover (the aggregator selects itself, seen in the smoke run at round 5) is allowed. The paper neither allows nor forbids it. | `selection.py` | Needs a decision (Q11). |
| R-20 | The liveness check raises `RuntimeError` from a task callback every 600 s. That is acceptable, but it currently does not count probes or selection in progress after this refactor. | `compass_simulation.py:250-260` | Must be updated with the protocol. |
| R-21 | `advertise_membership` puts f×1000 and B (as int) in a self-declared payload, and receivers store it in the view. | `community.py:451-452, 476` | Covered by A5. |
| R-22 | No test infrastructure: `pytest` is not installed. | — | Needs installing (I will ask before running pip). |

---

## 4. Paper issues (ambiguous, inconsistent, or problematic when implemented)

| ID | Issue | Proposed resolution (needs your OK where marked **Q**) |
|---|---|---|
| P-1 | **Who trains.** Sec. 3.4 has C ⊂ M\{a}, so the aggregator does not train, but this is never stated explicitly. **Q1** | The default is no training by the aggregator; `protocol.aggregator_trains` stays as an option. |
| P-2 | **Grace period vs. selection timing.** Q, f and B need late arrivals within the grace period, but selection for t+1 needs the updated view. Waiting out the grace period delays every round that has a straggler. **Q2** | (a) Aggregate at min(all, δ_dl), finish the state update at min(all, δ_dl + grace), then start selection (paper-faithful, slower). Or (b) start selection right after aggregation and fold in grace-period arrivals before the handover (the pool's Q may then be one round stale). I propose (a) as the default and (b) as an option. |
| P-3 | **B̂ = Z / T_up measures the path bottleneck**, not the client uplink. When n clients upload to one aggregator simultaneously, the aggregator's downlink is shared and B̂ underestimates B_i by up to a factor of n. The required test "estimates converge to truth" then fails. **Q3** | Model separate uplink and downlink per node. The default config makes the aggregator downlink non-binding (the paper's assumption that the uplink is the bottleneck). The test checks convergence under that assumption, and a second test documents the bias when the downlink binds. |
| P-4 | **"Without any additional messages"** vs. t_ack, which requires the client to send a download acknowledgement. | Treat the ack as a tiny control message, and count it in the bytes. |
| P-5 | **L is RTT** (Table 1, probing), but message delays are one-way. **Q4** | Use RTT everywhere in the objective (L_max is in RTT units). The one-way delay of a message is L/2 (+ noise). |
| P-6 | **"Resample with the next hash index"** is undefined. **Q5** | Use M^(t+1,k) = TopM H(i ‖ t+1 ‖ k) for k = 1, 2, … up to `max_resamples`, then fall back to self-serve. The current code takes the next M in the same order. |
| P-7 | **r_i under churn.** "Updated whenever a node joins" says nothing about leaves. Should departed nodes stay in r_i and N? **Q6** | Rows cover the current membership: on LEAVE, nodes drop that entry and N comes from the view's current membership. The alternative is all nodes ever joined, which is literal but stale. |
| P-8 | **The network-sum estimator** (N/M)·Σ_pool r_j assumes the pool is uniform. Pool members that do not respond shrink it, so should M be the nominal or the responding count? Also, the claim that "estimation error rescales all utilities by a common factor" holds only for the multiplicative form and only while the clip at 1 does not bind, because a noisy estimate can push Φ_repr above 1. | Use M_resp, the number of responders, in the denominator. Log how often Φ_repr is clipped at 1. Note that with the additive form the winner *can* depend on the estimate. |
| P-9 | **Epochs vs. runtime.** E full epochs of CIFAR on CPU for every client is slow for development. **Q7** | Train E real epochs. Add an optional `train.max_batches_per_epoch` cap, flagged in the logs as a deviation, for smoke runs only. Simulated time always follows E·D_i·c/f_i. |
| P-10 | **Baselines are client-selection methods**, so the aggregator choice is undefined for power-of-choice and Oort. **Q8** | Use the same hash pool. The aggregator is the eligible pool member with the smallest hash. Power-of-choice picks the top-n by local loss of θ^(t), where each pool member evaluates the loss on its data (adds simulated compute time). Oort-like ranks by (D_i·√mean loss²)·min(1, T_target/T_i)^α_oort using view estimates. Oracle_central uses the true f, B and L, every node as a possible aggregator, neighborhoods over all of V, and the configured solver with a larger budget. |
| P-11 | **DP noise vs. the S_ij/S_ji consistency check.** Independent noise in each direction makes honest nodes fail the check. Flooding noise also adds a small asymmetry. **Q9** | Tolerance = `ckks_tol + k·σ_dp·√2` (configurable). When flagged, drop the pair (use the default value and log it). The alternative is to exclude the node. |
| P-12 | **Flooding noise in practice.** "A fresh encryption of zero with large noise" has no TenSEAL primitive. Real flooding (≈40 extra bits) needs a larger modulus than the "shallow parameters" the paper claims. | Add Enc_pk(0) via `encrypt_zero`, plus an encoded Gaussian noise vector of std `σ_flood` (configurable), after the e_1 mask. The benchmark reports error vs σ_flood. The paper text could say this is statistical flooding at a chosen σ. |
| P-13 | **Φ_coh with \|C\| = 1** (greedy construction) has no pairs. | Φ_coh({i}) = 1. Algorithm 1 seeds with pairs, so this occurs only in other solvers. |
| P-14 | **Verification is limited.** The aggregator can check the score but not that C_a ⊆ N_a, that the latencies are honest, or that C_a is optimal. | Implement score and latency-consistency checks (L < L_max, \|C\| = n, C ⊆ pool\{a}) and document the limits. |
| P-15 | **"A^(t+1) fails during round t+1"**: the paper does not define how A(t) learns this. | Add a `round_done` notice from A(t+1) to A(t) and a watchdog timeout = δ_dl + grace + selection budget. On timeout, take the next-ranked proposal. Clients of the failed round that already trained are dropped. |
| P-16 | **Eligibility uses B_i (upload)**, but an aggregator mostly needs download capacity for n models. | Keep it as in the paper. Mention it in the threats-to-validity section. |
| P-17 | **The weights require w_m > 0**, while the task asks for ablations that "turn a component off". | Validate w_m > 0 strictly. Ablations use a small weight (e.g. 1e-3), as the paper says. Only w_c may be 0. |
| P-18 | **Bootstrap:** M^(1)'s smallest-hash member starts at context collection, but it holds the initial view whose f and B are only class priors, and there are no similarities until the join protocol finishes. | Run the join-time HE for the initial population before t = 0, outside the clock, configurable. Then A^(0) starts. |

---

## 5. Proposed plan (Phase 2), in order

Each step comes with its unit tests, run before moving on. Commits are grouped by component.

0. **Infrastructure.**
   - Install `pytest` (asking first).
   - Add `tests/` and `conftest.py` (PYTHONPATH).
   - Add the YAML config loader with dotted overrides and validation. The config dataclasses replace the COMPASS argparse flags, and the legacy `--flags` stay accepted through a thin shim.
   - Add `run.py` and `configs/{default,smoke}.yaml` plus one preset per baseline and solver.
   - Add run metadata: `config.resolved.yaml`, the seed, `git rev-parse HEAD` and a dirty flag.
   - Add a single seed hierarchy (`numpy.SeedSequence` → partition, devices, latency, failures, training, solvers, HE noise), set `model_seed`, pass seeds to the Partitioner, and set `PYTHONHASHSEED` inside `run.py`.
1. **Pure core** (`dlsim/compass/core/`, no IPv8 and no asyncio):
   - `utility.py`: all Φ terms, both forms, clipping, validation, partial sets, and an incremental state.
   - `problem.py`: `SelectionProblem`, `Budget`, `SolverResult`.
   - `neighborhood.py`: k_nb, `<` L_max, ties by id, decline.
   - `verify.py`.
   - `view.py`: the population view with τ, Q, f, B, m, κ and membership, plus the Q and f/B update rules and class priors.
   - `estimators.py`.
   - `similarity.py`: rows, r_i, the network-sum estimate, the consistency check.
   - Tests 1, 4, 5 (estimator part), 6, 7.
2. **Solvers** (`dlsim/compass/solvers/`): the registry, `greedy_swap` (incremental), `exact` (B&B with a size guard), `tabu`, `simulated_annealing`, `genetic` (SPMA adapted), `lns` and `random`. Tests 2 and 3, including an optimality-gap report for `greedy_swap`.
3. **HE** (`dlsim/compass/he.py`, rewritten):
   - `plaintext` (direct) and `ckks` via `tenseal.sealapi`: K slots, × ŷ_j, ⌈log₂K⌉ rotations, × e_1, + Enc_pk(0) + σ_flood, decrypt slot 0.
   - The public key and Galois keys go in the public context.
   - Once-per-identity answering, a stored public ciphertext, both directions, and optional DP noise.
   - Add `scripts/bench_ckks.py`.
   - Test 11 and the mask-leak test.
4. **Simulation environment**:
   - Device classes (f, B_up, B_down, failure probability, priors) and per-node up/down limits in `BWScheduler`.
   - Latency models `clustered` and `file`, with noisy RTT probes.
   - The timing model T_cmp = E·D_i·c/f_i, and the simulator emits t_dispatch, t_ack, t_first and t_last.
   - Failure and churn processes, including targeted aggregator failure.
   - Fix `create_model` and add a small ResNet, CIFAR-100, a Shakespeare hook, and seeded Dirichlet.
   - Test 5 (convergence under the simulator).
5. **Protocol** (rework `community.py` and `selection.py` around the core, keeping the IPv8/EVA/bypass structure):
   - Context collection with r_i and the row only, plus the consistency check.
   - The view is authoritative only through the handover; no view goes to clients or candidates.
   - Eligibility from the view.
   - P-probe median RTT with cost accounting and a prefetch option.
   - The k_nb neighborhood and decline.
   - The solver registry.
   - Proposals with latencies, verification, and ranking.
   - The deadline δ_dl and grace period (P-2), D_i-weighted FedAvg, and paper state updates (τ for all selected clients).
   - Fault handling: watchdog, next-ranked, next hash index, self-serve.
   - Bootstrap from M^(1), the hash-seed mode, and `aggregator_trains=false` by default.
   - Tests 8, 9 and 10 (two runs give identical `rounds.jsonl` decision fields).
6. **Selectors** (`dlsim/compass/selectors/`): `compass`, `plexus_random`, `power_of_choice`, `oort_like` and `oracle_central`, behind one interface and driven by config.
7. **Logging**: a per-round `rounds.jsonl` with every field in §H, plus the existing CSVs where cheap. Add `scripts/plot_compass.py` (accuracy vs sim time, Jain index, participation histogram, Φ traces, overhead).
8. **Docs and finish**:
   - A README section: architecture, adding a solver or selector, running a config, and a paper-parameter → config-key table.
   - Rewrite `dlsim/compass/COMMUNITY.md` to match.
   - Run `python run.py --config configs/smoke.yaml`.
   - Write a final change summary.

### Decisions taken (supersede sections 2–5 where they differ)

| Q | Decision |
|---|---|
| Q1 | The aggregator does not train. `protocol.aggregator_trains` defaults to false; the flag is kept. |
| Q2 | **Resolved.** Selection for t+1 starts as soon as round t closes, without waiting for the grace period. Candidates and the verifier use one immutable view snapshot, taken when the context is assembled. Late-arrival records are applied after the snapshot and never delay the handover. If they arrive after the handover, A(t) forwards them to A(t+1), which applies them on receipt. Option (a), waiting for the grace period, is available only as `closing.wait_grace_before_selection`. |
| Q3 | **Resolved.** Separate uplink and downlink for every node, and fair-share contention on the aggregator's ingress link that affects round time. A sample is censored when the aggregator's ingress rate reaches ρ·B_down(a), with ρ = 0.9 by default. A censored sample skips the B update but still updates f, and the censored fraction is logged. "Large aggregator downlink" is an option only. |
| Q4 | **Resolved.** L is the RTT everywhere (objective, neighborhood, L_max, probing). The one-way delay L/2 is used only for message delivery. |
| Q5 | M^(t+1,k) = TopM H(i ‖ t+1 ‖ k) for k = 1..`fault.max_resamples` (default 2). After that, A(t) serves the round itself. k is sent in the selection request so every pool member can verify its membership. |
| Q6 | Rows, r_i and N cover the current membership only. A node that fails to respond in `fault.max_missed` (default 3) consecutive selections or rounds is removed from the membership by the aggregator that observes it, and the other nodes drop its entries. Every membership change is logged. |
| Q7 | E real epochs. `train.max_batches_per_epoch` exists for smoke runs and tests only and is logged as a deviation. Simulated compute time always follows E·D_i·c/f_i. |
| Q8 | There are six baselines. |
| | `plexus`: a **port of the original Plexus sampling and aggregation logic** (`dlsim/plexus/`) into the shared simulator, with a sanity check against the original Plexus simulation (same participants per round for the same seed on a small no-churn config). |
| | `compass_random`: the full COMPASS protocol with the `random` solver. |
| | `power_of_choice`: the candidate set is the pool, and the loss evaluation costs simulated time and messages. |
| | `oort_like`: as proposed in P-10, with every simplification relative to Oort documented in the README. |
| | `oracle_central`: only the selection step is free. Training, transfers and round closing are simulated as usual, and the candidate-aggregator set is configurable (`all` or top-K by f and B). |
| Q9 | Always use (S_ij + S_ji)/2, with no default fallback. If \|S_ij − S_ji\| > ckks_tol + k·σ_dp·√2 (k = 3), still use the average, log the flag and increment a per-node flag count. Excluding flagged nodes is an option, off by default. |
| Q10 | True per-node f and B are sampled **directly from the FedScale profiles**. Device classes come from quantile cut-points on a combined compute/bandwidth score (3 classes by default, cut-points configurable). The priors f̄ and B̄ are the class medians of the sampled values, and σ̄_κ = relative spread × the class mean of μ. FedScale bandwidth is the upload rate, and download = ratio × upload (ratio 4 by default, documented as an assumption). On top of the true values come a per-round log-normal fluctuation (CV per class) and rare sudden slowdowns (probability and factor configurable, e.g. 0.02 and 3–5×). Per-class dropout probabilities are configurable. |
| Q11 | A(t) may be selected again as A(t+1) or as a client of A(t+1); the optional `forbid_*` flags are off by default. Log how often the same node aggregates consecutive rounds. |
| Q12 | The client-side backup aggregators are kept as `fault.client_backups`, off by default. The paper's watchdog plus next-ranked fallback is the default. |
| Q13 | Done: `pytest` 8.4.2 is installed. |
| Q14 | `python run.py --config ... key=value` is primary. `simulations/compass/*.py` stay as thin wrappers. The original Plexus, DL and GL simulations stay untouched. |

These decisions replace the following earlier content:

- R-7 and P-1 (by Q1).
- P-2 (by Q2).
- P-3 (by Q3).
- P-5 (by Q4).
- P-6 (by Q5).
- P-7 (by Q6).
- P-9 (by Q7).
- E2, E5 and P-10 (by Q8).
- P-11 and R-6 (by Q9: no default S value is used anywhere).
- G2 and G2-env (by Q10).

---

## 6. Revision 2 (paper and task revised after the first audit)

Sources: the revised `main.tex` (Sec. 3.3 "Completion-time prediction"; Sec. 4.7 "Predictive round closing", "State updates", "Timing of state updates"; "Straggler handling" in Related Work) and `CLAUDE_CODE_TASK.md` revision 2.

### 6.1 New and changed gap rows

None of these exist in the code today, so every status is relative to the current code.

| # | Item | Status | Where | What must change |
|---|---|---|---|---|
| A5′ | The population view also holds σ_i, plus a missed-response counter (Q6) | missing | `peer_manager.py:19-25` | Add σ_i (initialized to σ̄_κ) and `miss_count`, both written only by aggregators. |
| A-RTT | Latency convention: L = RTT everywhere, and L/2 only for delivery | wrong | `caches.py:55` (RTT/2 stored), `learning_simulation.py:225` (matrix value used as the one-way delay) | Store and compare RTTs. The simulated message delay is L/2. See P2-6 for the matrix itself. |
| A7′ | Q_i ← (1−α)Q_i + α·exp(−γ·ℓ_i), with ℓ_i = max(0, t_last − d_i) and ℓ_i = ∞ beyond d_i + g; Q(0) = 1 | wrong | `community.py:1115-1137` | This replaces row A7: lateness is measured against the client's own deadline, not a global target. |
| A8′ | Censored bandwidth samples: if the aggregator's ingress rate during the upload was ≥ ρ·B_down(a), skip the B update and still update f; log the censored fraction | missing | — | The BW scheduler must expose the aggregator's ingress rate over each transfer's lifetime, computed as a time-weighted maximum or mean (P2-5). |
| A9 | Completion-time prediction μ_i = E·D_i·c/f_i + Z/B_i; δ̂_i = t_last − t_ack; σ_i² ← (1−β_i)σ_i² + β_i(δ̂_i − μ_i)², using μ before f and B are updated; σ initialized to σ̄_κ | missing | — | Add it to `core/estimators.py`. The order within one record is Q, then σ, then f and B, then m += 1. |
| C-close | `RoundClosingPolicy` interface with four modes. | missing | `community.py:998-1063` (success_fraction + timeout) | New `dlsim/compass/closing/` package. Straggler updates are never aggregated, but their timing is recorded within d_i + g. |
| | `predictive` (default): d_i = t_ack_i + μ_i + z·σ_i. A client is a straggler if its update is not complete by d_i, or if it has no ack within δ_dl of dispatch. The round closes when every client has delivered or been declared a straggler, and never later than dispatch + δ_dl. | | | |
| | `wait_all`: wait for every client, still capped at δ_dl. | | | |
| | `fixed_deadline`: a single deadline δ_dl for everyone. | | | |
| | `over_selection`: select ⌈(1+o)·n⌉ clients and close when the first n have arrived, or at δ_dl. | | | |
| C-refl | Config hook for REFL-style merging of late updates into the next round with reduced weight | missing | — | Add the hook (`closing.late_merge`), off by default. Only the interface is needed. |
| C-snap | Timing of state updates: an immutable snapshot at context assembly; late records applied afterwards; forwarded to A(t+1) if they arrive after the handover; option (a) behind a flag | missing | — | Use a copy-on-write view snapshot with an id. The context and proposals carry the snapshot id. The verifier uses only the snapshot. Add a `late_records` message type. |
| C-state′ | State update order: τ ← t for every selected client, then Q, then σ, then f and B | wrong | `community.py:1102-1137` | Replaces row C16. |
| G-bw | Per-node upload and download bandwidth, and fair-share contention on the aggregator's ingress link that affects round time | partial | `bandwidth_scheduler.py:17, 78-120` | The scheduler has one `bw_limit` for both directions. It allocates first-come, sender-limited min(sender_left, receiver_left) instead of max-min fair sharing, and does not re-allocate when a transfer finishes. Split it into `bw_up` and `bw_down` and implement max-min fair re-allocation on every arrival and departure. |
| G-fluct | Per-round multiplicative log-normal fluctuation of the true f and B (CV per class), plus sudden-slowdown events | missing | — | Add it to `env/devices.py`, drawn from the seeded RNG once per (node, round). |
| G2-env | Environment data | partial | `data/fedscale_traces`, `data/latencies.txt`, `learning_simulation.py:136-157, 206-228` | See 6.2. |
| H′ | Per-round log also has the closing policy, each client's d_i and actual t_last, the censored fraction, membership changes, and the consecutive-aggregator count | missing | — | Add the fields to `rounds.jsonl`. |
| I′ | Config also has z, g, ρ, the closing policy, o, the late-merge hook, the fluctuation and slowdown parameters, the download/upload ratio, the class cut-points, the city assignment and `max_missed` | missing | — | Add the keys. |
| T10–T14 | New tests: predictive closing (10), σ convergence and censoring (11), snapshot consistency (12), determinism (13), CKKS (14) | missing | — | See the plan in 6.5. |

### 6.2 Environment data already in the repository (G2)

| File | Content (checked) | Use |
|---|---|---|
| `data/fedscale_traces` | A pickle dict of **500,000** devices (not 131k), each with `computation` (ms per sample, range 15–199, median 41) and `communication` (kbps, range 1,017–204,057, median ≈ 6,075). There is **no availability data and no network type.** | True f per node = c·1000/ms per sample. With c configurable (default 1), f is in samples/s and f_ref, f_min and f̄ use the same units. B_up = kbps·1024/8 bytes/s, as Plexus computes it, and B_down = ratio × B_up. Devices are sampled with the seeded RNG, fixing R-9 (compute and bandwidth are taken from the same device). |
| `data/latencies.txt` | A **227×227** matrix in ms (WonderNetwork average ping, i.e. RTT). It has **51 negative entries** (min −0.5 ms), an **asymmetric** part (median relative asymmetry 0.2%, but 2.3% of pairs differ by > 20%), and diagonal entries around 0.02 ms. | City-level RTT, combined as described in P2-6. |
| FedScale availability traces | **not present** (`--availability-traces` expects an external pickle) | Supported as an option if supplied. Churn otherwise comes from per-class dropout plus scripted failures. |

Additions needed:

- **City assignment.** Support `round_robin` (Plexus) and `concentrated` (a few regions with configurable weights).
- **L_max.** Either an absolute value or a percentile of the realized RTT distribution.
- **Access-network delay per class.** FedScale has no network type, so these must be configured per class (default values are listed in P2-6).

### 6.3 New bugs and risks

| ID | Issue | Where | Impact |
|---|---|---|---|
| R-23 | Plexus (and therefore the current COMPASS simulation) applies the matrix value, which is an RTT, as the **one-way** delay. Messages are delayed twice as long as the RTT convention implies. | `learning_simulation.py:225`, `bypass_network_community.py:47-49` | Timing differs from the Q4 convention. See P2-6 for its effect on the Plexus comparison. |
| R-24 | The BW scheduler is not max-min fair. A transfer gets min(sender_left, receiver_left) when it starts and keeps it until it finishes, even when other transfers end. | `bandwidth_scheduler.py:78-120` | Ingress contention and the censoring test (Q3) would be wrong. |
| R-25 | Negative matrix entries would give negative delays, and `int()` truncation turns them into 0. | `learning_simulation.py:225` | Clamp to ≥ 0 when loading. |
| R-26 | Out-of-order late records: a late record from round t can reach A(t+1) after A(t+1) has already updated the same node in round t+1 (if it was reselected). The EMA updates do not commute. | (new code) | Apply the records in round order, or skip stale ones (P2-7). |

### 6.4 New paper issues (revision 2)

| ID | Issue | Proposed resolution |
|---|---|---|
| P2-1 | **μ_i ignores ingress contention.** μ_i = E·D·c/f_i + Z/B_i assumes the client's uplink is the bottleneck. With Q3 (realistic aggregator downlink), n clients that finish training close together share B_down(a), and their upload takes up to n× longer than Z/B_i. Two consequences follow. Predictive closing then declares **healthy clients stragglers because of the aggregator's own congestion**. Their lateness also lowers Q (the "predictability" score), punishing them for something they did not cause. | Make the deadline contention-aware and keep the estimators paper-exact. The aggregator knows how many uploads it is receiving, so it could set d_i with Z / min(B_i, B_down(a)/k_i), where k_i is the number of concurrent uploads at t_first. Lateness of a censored upload (ingress saturated) would also not count against Q (ℓ_i := 0 if the update arrives by d_i + g). The alternative is to keep the paper formula and accept the bias. **Question RQ1.** |
| P2-2 | **Deadline beyond the cap.** If d_i > dispatch + δ_dl, the round closes at the cap and the client is excluded, yet an update that arrives at the cap ≤ t_last ≤ d_i has ℓ_i = 0, so Q does not register the exclusion. | Use the effective deadline d_i′ = min(d_i, dispatch + δ_dl) for both straggler status and lateness. **RQ2.** |
| P2-3 | **Lateness under the other policies.** Under `wait_all`, `fixed_deadline` and `over_selection` the paper defines no d_i, yet Q needs ℓ_i. With `over_selection`, it is also unclear whether the solver selects ⌈(1+o)n⌉ (with Φ over that set, and k_nb ≥ ⌈(1+o)n⌉ required) or selects n and pads. | Always compute the predictive d_i and use it for ℓ_i, Q, σ and the logs, so the state dynamics stay comparable across policies. The policy decides only when the round closes and who is aggregated. For `over_selection`, the solver is run with n′ = ⌈(1+o)n⌉, and every one of the n′ gets τ ← t. **RQ3.** |
| P2-4 | **No acknowledgement.** A client that does not acknowledge within δ_dl has no t_ack, so d_i is undefined. If its update still arrives later, should anything be recorded? | ℓ_i = ∞ and no measurement (no t_ack means T_cmp cannot be computed). It is a straggler at dispatch + δ_dl. **RQ4** (a low-stakes confirmation). |
| P2-5 | **Censoring and σ.** The paper says censored samples skip B but update f. It is silent on σ, whose sample (δ̂ − μ)² includes the contention delay, which would inflate σ_i and loosen future deadlines. The censoring rule also needs a precise "rate during the upload" (instantaneous maximum or time-average over [t_first, t_last]). | Skip the σ update for censored samples. Censor if the **time-averaged** ingress utilization over [t_first, t_last] is ≥ ρ. **RQ5.** |
| P2-6 | **Latency data and the Plexus comparison.** The WonderNetwork values are RTTs, but Plexus uses them as one-way delays (R-23). Under Q4 our `plexus` port therefore sees half the per-message delay of the original Plexus simulation, so "like-for-like" holds for participants but **not for round times**. The matrix is also asymmetric and has negative entries. | L_ij = max(0, (W_ij + W_ji)/2) for the cities of i and j, plus access(κ_i) + access(κ_j) + jitter, where jitter is log-normal with a configurable CV and is redrawn for each probe and each message. The access-delay defaults are 5, 15 and 30 ms per class, configurable. All baselines, `plexus` included, use the same convention. The sanity check against the original Plexus compares the participants per round, not the timing, and the README documents the difference. The alternative is a `latency.plexus_compat` flag that applies the matrix value as the one-way delay. **RQ6.** |
| P2-7 | **Late records forwarded after the handover.** If A(t+1) has itself already handed over, or has already updated the same node in round t+1, the forwarding chain and the ordering are unspecified. | Records carry (node, round). A holder forwards them along the chain to the current aggregator, with at most one forward per hop and deduplication. A record older than the node's last applied round is applied to Q, σ and f/B only if no newer record for that node has been applied; otherwise it is dropped and logged. |
| P2-8 | **Units.** FedScale gives ms/sample, not CPU cycles, so c is not identifiable. | Set c configurable (default 1), so f is in samples/s, and document it. Only ratios such as f/f_ref and T_cmp matter. |
| P2-9 | **Persisting `max_missed` (Q6) across aggregators.** The counter must live in the view, because any aggregator may observe a miss. | Add `miss_count`, written only by aggregators. It is reset on any successful response. A node is removed after `max_missed` consecutive misses, and the removal travels with the view. Removal also triggers the nodes to drop the row entry, through a LEAVE notice from the aggregator. |

### 6.5 Plan changes (revision 2), merged into section 5

- **Step 1 (core):**
  - Add σ_i and `miss_count` to the view.
  - In `estimators.py`, add μ, the σ update, the update order (Q → σ → f, B → m) and the censoring flag.
  - Add a copy-on-write `ViewSnapshot` with an id, plus `LateRecord` application with ordering (P2-7).
  - Tests 4 and 11 (σ convergence, censoring leaves B unchanged), and test 12 (snapshot consistency).
- **Step 4 (environment):**
  - `env/devices.py`: FedScale sampling, quantile classes, priors, σ̄, the down/up ratio, per-round log-normal fluctuation, slowdowns and per-class dropout.
  - `env/latency.py`: the WonderNetwork RTT with symmetrization and clamping, access delays, jitter, `round_robin` and `concentrated` city assignment, and L_max as an absolute value or a percentile.
  - `BWScheduler` with up/down limits, max-min fair re-allocation, and an ingress-utilization trace per transfer.
  - Scripted failures, including an aggregator failure at round r.
  - Test 5, now under fluctuation and contention, and a fair-sharing test.
- **New step 4b (closing):** the `RoundClosingPolicy` interface with `predictive`, `wait_all`, `fixed_deadline` and `over_selection`, the grace handling, and the `late_merge` hook (off). Test 10 (no false stragglers with deterministic speeds and z > 0, an injected slowdown is cut at d_i, and z → ∞ behaves like `fixed_deadline`) and test 8.
- **Step 5 (protocol):**
  - Selection starts at round close, using the snapshot.
  - Late records after the snapshot, and the forwarding message.
  - k in the selection request (Q5).
  - Membership removal after `max_missed` (Q6).
  - S averaging and flags (Q9).
- **Step 6 (selectors):** `plexus` (a port of `dlsim/plexus` logic, plus the sanity check), `compass_random`, `power_of_choice`, `oort_like` and `oracle_central` (candidate set `all` or `topk`).
- **Step 7 (logging):** add the closing fields, the censored fraction, membership changes and consecutive-aggregator counts.

### 6.6 Open questions on revision 2 (proposed default in brackets)

- **RQ1:** Should the deadline be contention-aware, and should censored uploads not count as late for Q? [yes to both; the estimators stay paper-exact]
- **RQ2:** Should the effective deadline be d_i′ = min(d_i, dispatch + δ_dl) for both straggler status and lateness? [yes]
- **RQ3:** Should the predictive d_i always be used for ℓ, Q and σ under every closing policy, and should `over_selection` solve with n′ = ⌈(1+o)n⌉? [yes, yes]
- **RQ4:** A client with no ack within δ_dl: ℓ = ∞ and no measurement? [yes]
- **RQ5:** Should censored samples also skip the σ update, and should censoring use the time-averaged ingress utilization over the upload? [yes, time-averaged]
- **RQ6:** Latency: symmetrize and clamp the matrix, add per-class access delay and jitter, and use one-way = L/2 for every baseline including `plexus` (so round times differ from the original Plexus simulation), with an optional `plexus_compat` flag? [yes]

### 6.7 Final decisions on revision 2 (approved, Phase 2 started)

- **RQ1 (adjusted):**
  - The contention-aware deadline estimates k_i as the number of clients in the round whose **predicted** upload windows [t_ack_j + E·D_j·c/f_j, that + Z/B_j] overlap client i's own window. The upload term is Z / min(B_i, B_down(a)/k_i).
  - Every d_i is recomputed whenever a new acknowledgement arrives.
  - Uploads censored as congested never count as late for Q.
  - The estimators stay paper-exact.
- **RQ2:** the effective deadline is d_i′ = min(d_i, dispatch + δ_dl), used for both straggler status and lateness.
- **RQ3:**
  - The predictive d_i is always used for ℓ, Q and σ, whatever the closing policy.
  - `over_selection` solves with n′ = ⌈(1+o)n⌉, and all n′ clients get τ ← t.
- **RQ4:** a client with no acknowledgement within δ_dl has ℓ = ∞ and no measurement, and is a straggler at dispatch + δ_dl.
- **RQ5:**
  - Censored samples also skip the σ update.
  - Censoring uses the time-averaged ingress utilization over [t_first, t_last], with the threshold at ρ.
- **RQ6 (adjusted):**
  - **No clamping.** For each pair: if both directions are valid (> 0), use their mean; if only one is valid, use it; if neither is, the pair is invalid.
  - Cities that still have an invalid pair are excluded from node assignment.
  - The excluded cities and the cleaning statistics (entries averaged, entries taken from one direction, pairs invalid, cities excluded) are logged per run.
  - Access delay per class defaults to 5 / 15 / 30 ms and is configurable.
  - Jitter is log-normal.
  - The one-way delay L/2 applies to all baselines.
  - `latency.plexus_compat` is optional.
  - The Plexus sanity check compares participants per round.
- **P2-7 (adjusted):**
  - Every late record is tagged with its round index.
  - When a record is applied while the view already holds a newer update for that node (from a later round), the stale record is discarded and logged; otherwise it is applied.
  - EMA updates are never reordered.
- **FedScale logging:** each run logs the number of FedScale devices in the trace, the number sampled, and any filtering applied.

---

## 7. Phase 2 status (implementation complete)

Every checklist item in sections 2 and 6.1 is now `done`. Where each one lives:

| Area | Status | Where |
|---|---|---|
| A. Node state, population view (σ, m, κ, miss counter, flags), RTT convention, Q with lateness, f/B/σ estimators, censoring, update order | done | `core/view.py`, `core/estimators.py`, `closing.py`, `community.py` |
| B. Utility (all Φ, both forms, clipping, validation, partial sets, single pure function, incremental state) | done | `core/utility.py`, `config.py` |
| C. Hash pool (+ model-hash seed, resample index k verified by members) | done | `core/hashing.py`, `community.py` |
| C. Bootstrap A^(0) = argmin hash of M^(1) | done | `env/world.py:start` |
| C. Context collection (r_i and row only), averaging and consistency flags, snapshot | done | `core/similarity.py`, `community.py:build_context` |
| C. Eligibility from the snapshot | done | `community.py:build_context` |
| C. Probing (P probes, median RTT, simulated cost, prefetch) | done | `community.py:probe`, `on_prefetch` |
| C. Neighborhood, local solve, proposal, verification, ranking | done | `core/neighborhood.py`, `core/context.py`, `community.py` |
| C. Handover, watchdog, next-ranked fallback, resample, self-serve; client backups (option) | done | `community.py:select_and_handover`, `handover_chain` |
| C. E epochs, D_i returned; RoundClosingPolicy (4 modes); late records; snapshot timing; REFL hook; D-weighted FedAvg | done | `closing.py`, `aggregation.py`, `community.py` |
| D. Solver interface, registry, budget; `greedy_swap`, `exact`, `tabu`, `simulated_annealing`, `genetic`, `lns`, `random` | done | `solvers/` |
| E. compass, compass_random, plexus (original logic ported), power_of_choice, oort_like, oracle_central | done | `selectors.py`, `simulations/compass/plexus_shared.py` |
| F. plaintext / ckks (paper protocol), both directions at join, once per identity, DP noise, benchmark, mask test | done | `he.py`, `env/world.py`, `scripts/bench_ckks.py`, `tests/test_he.py` |
| G/G2. Clock, FedScale devices and classes, WonderNetwork RTT cleaning and city assignment, up/down links with fair sharing, fluctuation and slowdowns, churn (dropout, crashes, scripted failures, joins, availability traces), CIFAR-10/100, FEMNIST/Shakespeare hooks, models | done | `env/`, `data.py` |
| H. Structured logs, run metadata, plots | done | `runlog.py`, `env/world.py`, `scripts/plot_compass.py` |
| I. YAML config with dotted overrides, presets, `run.py` | done | `config.py`, `configs/`, `run.py` |
| Tests 1–14 (and more: 129 tests) | done | `tests/` |

### 7.1 Findings made during implementation (for the paper)

| ID | Finding | What the code does |
|---|---|---|
| I-1 | **Incremental swap formula.** P' = P − σ_u + σ_v − S_uv holds only if σ_u excludes the diagonal term S_uu of the member u (σ_u = Σ_{j∈C, j≠u} S_uj). With σ_v = Σ_{j∈C} S_vj taken literally for every neighbor, P' is off by 1. | σ includes S_uu for members, and the code subtracts it explicitly (`core/utility.py`). Test 2 checks this against full recomputation. |
| I-2 | **Single point of failure when A(t+1) = A(t).** If A(t) selects itself (or serves the round itself) and then crashes, no other node holds θ or the view, so training stops. The fault handling of Sec. 4.6 assumes A(t) ≠ A(t+1). | Allowed by default (Q11). `protocol.forbid_repeat_aggregator=true` removes the gap, except for the self-serve last resort. A possible fix is for A(t) to keep its predecessor's watch, or to hand a copy to the best other candidate. |
| I-3 | **Prefetching needs a trigger.** Pool members "can compute M^(t+1) in advance", but they do not know when round t runs. | A(t) sends M small `prefetch` notices at dispatch. They are counted in the probe traffic. |
| I-4 | **Plexus liveness under realistic timings.** The original Plexus code needs ≥ 3 models within `aggregation_timeout`, or it drops the aggregation and restarts. With FedScale compute times and E = 1 epoch on ~3,100 samples, completion times range from about 47 s to 620 s, and the original logic stalls at round 1 (seen in the smoke config with 300 s, and still with 600 s). | `baselines.plexus.aggregation_timeout` defaults to δ_dl. A liveness guard stops stalled runs with an error. Comparisons with Plexus must state the timeout used. |
| I-5 | **Membership knowledge differs between nodes under churn**, so a member may not find itself in the pool it is asked about. | The request is still answered, and the mismatch is logged as `pool_membership_mismatch`. |
| I-6 | **power_of_choice costs a model download** for each of the M pool members per round. This is inherent to the method in a serverless setting. | Simulated and counted under the traffic category `poc`. |
| I-7 | **A crash right after acknowledging a handover can lose the acknowledgement**, because it is still in flight. | Treated as no acknowledgement: A(t) falls back, which is safe. |

### 7.2 Remaining items and limitations (not checklist items)

- **Simulation only.** The new protocol runs on the simulated transport (`env/world.py`). A real IPv8/EVA transport for it
  was not part of this task.
- **CKKS costs no simulated time.** Join-time CKKS operations take no simulated time; their wall time and byte sizes are logged.
- **CIFAR-100 is not downloaded.** It needs `data.download=true` (CIFAR-100 is not in `~/dfl-data`), and was not run.
- **The Plexus port** supports the datasets of the original code (CIFAR-10, FEMNIST).

### 7.3 Consolidation (after Phase 2, at the author's request)

The first implementation (old `community.py` logic, `fitness.py`, `selection.py`, `sample_manager.py`,
`peer_manager.py`, `caches.py`, `payloads.py`, the zero-sum-mask `he.py`, `COMMUNITY.md`, `compass_simulation.py`, the
old bypass community) and its edits to shared Plexus files (datasets' `label_counts`, the `Partitioner` change, the
COMPASS flags in `simulations/args.py`, the `learning_simulation.py` edits) were removed. This replaces decision Q14:
there are no legacy wrappers any more, and `run.py` + YAML is the only entry point. `spma.py` moved to
`dlsim/compass/solvers/spma.py`, where it is the engine of the `genetic` solver.

### 7.4 Straggler calibration (2026-10-13, approved by the author)

Diagnosis (`experiments/straggler_diagnosis.py`): first-time participants were the main source of stragglers (28–31%
against 8–12% for repeat clients), because the class priors are wide (true f / prior of a class spans 0.4–4.9x in the
FedScale sample) while sigma_bar was a fixed 0.2 x mean mu. Separately, every silent client (dropout) stretched its
round to delta_dl, because the paper declares a missing acknowledgement only at dispatch + delta_dl.

New defaults, compared in `experiments/results/straggler_fixes` (N=100, 60 rounds, 2 seeds, no SGD):

| variant | stragglers | first-time | repeat | mean round | p90 round | updates/round | time for 60 rounds |
|---|---|---|---|---|---|---|---|
| previous defaults | 14.5% | 30.6% | 12.0% | 117 s | 600 s | 8.55 | 7099 s |
| sigma_prior = class_rmse | 8.0% | 10.9% | 7.6% | 136 s | 600 s | 9.20 | 8208 s |
| 5 device classes | 14.8% | 35.9% | 11.6% | 124 s | 600 s | 8.52 | 7474 s |
| ack_factor = 3 | 15.1% | 27.2% | 13.1% | 45 s | 74 s | 8.49 | 2771 s |
| **class_rmse + ack (new default)** | **7.5%** | **10.0%** | **7.1%** | **55 s** | **87 s** | **9.25** | **3371 s** |

- `env.devices.sigma_prior = class_rmse`: sigma_bar_k = sqrt(RMSE of the prior prediction of mu inside class k^2 +
  (CV_k x mean mu)^2). Paper change: the definition of the nominal sigma_bar_k.
- `closing.ack_factor = 3`, `ack_margin = 10 s`: a client without an acknowledgement by dispatch + 3 x predicted
  download time + 10 s is a straggler (lateness inf). Paper change: the "no acknowledgement within delta_dl" rule.
- Five classes did not help and are not used (3 classes stay the default).

### 7.5 No self-declared capacity; capped dataset size (2026-10-13, approved by the author)

Nodes no longer declare a device class. Capacity, bandwidth, sigma, staleness and reliability all come from
measurements; before its first measurement every node starts from one public prior (median f and B_up of the whole
device trace, sigma_bar from the class_rmse rule computed over all devices). A lie about the class could only last one
participation anyway (beta_i = 1 at m = 0), but removing the declaration removes a trust assumption from the paper.
The only remaining declared value is the dataset size, which cannot be measured (the compute time depends on D/f). Every
receiver caps it at a public D_max (default: the 95th percentile of the dataset sizes; 5 of 100 nodes are capped at
alpha = 0.3, losing 2.6% of the total weight), and the capped value is used everywhere: FedAvg weights, mu, f_hat.

- `protocol.declared_class = false` (true restores per-class priors), `protocol.D_max = 0` (absolute cap; 0 = derived),
  `protocol.D_max_quantile = 0.95` (>= 1 disables the cap).
- Cost, same setup as 7.4 (N=100, 60 rounds, 2 seeds, no SGD):

| variant | stragglers | first-time | repeat | mean round | p90 round | updates/round | time for 60 rounds |
|---|---|---|---|---|---|---|---|
| declared classes (7.4 default) | 7.5% | 10.0% | 7.1% | 55 s | 87 s | 9.25 | 3371 s |
| **public prior + D_max (new default)** | **7.8%** | **9.0%** | **7.6%** | **58 s** | 123 s | **9.22** | **3536 s** |

- Paper changes: system model (D_i is the capped declared size, nothing else is declared), notation table (public
  priors f_bar, B_bar, sigma_bar; D_max), initialization paragraph, the sigma_bar equation (over all devices), the
  population-view paragraph, the security discussion and the limitations.


### 7.6 Phi_strag renamed to deadline adherence Phi_adh (2026-10-13, approved by the author)

Q_i measures lateness against the node's own predicted deadline (mu_i + z sigma_i), not speed: predictable slowness is
already penalized by Phi_comp. The term is therefore called deadline adherence. Config key `utility.weights.adh`
(`weights.strag` in old YAML files, overrides and run logs is still accepted and renamed on load); the logged component
is `adh`. Paper: Phi_adh, w_adh, "deadline-adherence score Q_i", label `eq:adherence_score`, and one sentence that
contrasts Phi_adh with Phi_comp. "Straggler" still names a client that misses its deadline.

### 7.7 Baselines for the paper (2026-10-14, agreed with the author)

Main comparison (experiments/sweeps/main.yaml, main_femnist.yaml): compass, compass_random, plexus (n = 10, success
fraction 0.8), plexus_paper (s = 13, 0.8), fl_server (new: the FL reference of the Plexus paper), power_of_choice,
oort_like, fedcs (new: FedCS, Nishio & Yonetani 2019) and oracle_central. GL and D-PSGD are not ported: they are a
different paradigm (every node trains every round) and the Plexus paper already compares them with Plexus.
The Plexus port now applies the per-round device dropout of the shared environment (`baselines.plexus.dropout`,
default true), which the original simulation lacks; without it Plexus would be compared under easier conditions.
E2 (factorial.yaml): selection {informed, random} x closing {predictive, fixed deadline}.

