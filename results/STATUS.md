# line_follower multi-track: session state (2026-08-26, ~00:40 EDT)

## Where the task stands
Best checkpoint: `models/line_follower_multi/multitrack_sweep_w3_300k.zip`
**83.3% (20/24)** under per-reset track randomization, deterministic eval.
Not solved (bar is effectively all agents). Progression this session:
4.2% (single-track policy) -> 12.5% (prior continuation) -> 83.3%.

Winning config = PPO grid index 3: lr=3e-4, ent_coef=0.0, n_steps=256,
batch_size=512, gamma=0.99. Largest rollout/batch in the grid; the trials
that collapsed mid-run all had smaller rollouts or higher lr.

## RESULT: all four continuations REGRESSED (2026-08-26 ~01:10)
500k more steps at lr=3e-4 from the 83.3% checkpoint, four independent seeds:

| run | solve rate | training mean_ep_reward |
|-----|-----------|-------------------------|
| multitrack_sweep_w3_300k (start) | **83.3%** | 147.3 |
| cont_seed1 | 60.0% | 108.6 |
| cont_seed0 | 27.5% | 43.8 |
| cont_seed2 | 0.0% | 68.0 |
| cont_seed3 | 0.0% | 67.7 |

Four independent seeds all regressed, so this is NOT bad luck -- it is the
same peak-then-destroy pattern the Phase 2 DR arc hit, now reproduced under
controlled seeds. Note fresh optimizer state does NOT prevent it (--init-from
resets the optimizer, and these still collapsed), which argues AGAINST the
old "accumulated optimizer fragility" hypothesis. The likely cause is simply
lr=3e-4 being far too high to fine-tune an already-competent policy, with
ent_coef=0 offering no protection.

Training reward tracked the collapse (147 -> 43-108), so it IS a usable early
warning here even though its absolute value understates solve rate.

## PUBLISHED 2026-08-26 07:06 (Lucas's call: bank something usable)
Clean, uncontended 40-episode eval of the best checkpoint: **87.5% (35/40)**.
(The earlier 83.3% was measured WHILE the four continuations were training --
concurrency costs a few points even with the topic fix, so publish numbers
must be measured on an idle machine.)

Pushed to `CursedRock17/gazebo-gymnasium-policies` as a NEW folder
**`line_follower_multitrack/`** (model.zip + an honest model card saying it is
NOT solved, and warning against continuing it at lr=3e-4). The existing
`line_follower/` folder -- the fully-solved single-track policy -- was left
untouched on purpose: overwriting it would have been a regression for that
task. Verified by re-downloading: obs (12,64,64), act (2,).

## RUNNING NOW: phase 3 (detached, `results/phase3.sh`)
Four low-lr fine-tunes from the 87.5% checkpoint, 250k steps each, with
`--checkpoint-every 25000`: lr=5e-5 seeds 0/1 and lr=2e-5 seeds 0/1. The
script then evaluates EVERY snapshot (not just final models) into
`results/phase3_eval.txt` and appends the top 3. Read that file first.

Failures now skew to the first steps after reset (several terminate at step
1-2), so if low-lr fine-tuning plateaus, look at spawn pose / first-frame
handling rather than cornering or more steps.

## Next step when picking this up
Best checkpoint is still `multitrack_sweep_w3_300k.zip` at 83.3%. Continue
from it with a MUCH lower learning rate (`--lr 5e-5` or lower) and
`--checkpoint-every`, then eval the snapshots and take the peak rather than
the final model. Both flags were added for exactly this. If low-lr
fine-tuning also plateaus near 83%, stop adding steps -- the remaining
failures are concentrated at reset, so the lever becomes reward/DR shaping
or spawn-pose handling, not more training.

## Landed this session (lint-clean, uncommitted per the standing gate)
- **Camera topic collision fix** (`inprocess_vec_env.py`): topics are now
  `/rl/<pid>_<uuid8>/camera_<i>`. Unnamespaced topics made concurrent camera
  envs read each other's frames -- silent, and worth 95.8% -> 0% on a
  known-good checkpoint. Regression test in `test_line_follower.py`.
- **`_await_frames` hardening**: timeout scales with agent count; stale-frame
  fallbacks are counted (`stale_frame_waits`) and warned instead of silent.
- **sweep.py**: `--jobs` (parallel trials), `--track-shapes`, `--init-from`
  (weight transfer, not load(**cfg) -- that silently drops swept lr/n_steps),
  `--only-trial`, `--best-out` (concurrent sweeps clobbered one winner file).
- **train.py**: `--track-shapes`, writing to its own models/ dir.
- **`training_scripts/dr_eval.py`**: per-agent solve-rate eval promoted from
  a scratchpad (deploy.py's `dones.all()` loop misreports once agents desync).

## Measured, decided, do not re-derive
- More agents per world is SLOWER: 316/251/151/124 agent-steps/s at N=4/8/16/32.
  Optimum is ~6 processes x 4 agents (~494), K=8 regresses.
- GPU rendering is a DEAD END for speed. Getting ogre2 genuinely onto the
  NVIDIA GPU (system GL ahead of conda libglvnd + NVIDIA EGL vendor JSON)
  changed throughput 128.0 -> 126.6 agent-steps/s, i.e. nothing.
- The camera pipeline is 25 of 28ms per server step (physics is 3.2ms; track
  collision geometry only 6%). The cost is per-frame overhead, not pixels.
- **2.3x speedup: REAL, TRIED, REVERTED -- needs one more fix to land.**
  The server renders ~2.4 frames per step and the policy reads one.
  update_rate 30 -> 12.5Hz cuts server step 29.0 -> 12.5ms, and solve rate is
  unaffected (known-good checkpoint scored 95.8%, identical to its 30Hz
  baseline). 12.5Hz is also closer to the physical camera's 15 fps.
  BUT adopting it made every RESET slow: `_post_reset_settle` demands TWO new
  frames while advancing only 8 ticks (80ms), which is exactly ONE frame at
  12.5Hz, so each reset burns the full frame-wait timeout. My 2.3x was
  measured on `_step_server` in isolation and never exercised a reset.
  Reverted to 30Hz to keep the tree in a verified-good state.
  **To land it:** make `_post_reset_settle` advance 16 ticks (2 camera periods
  at 12.5Hz), then re-validate BOTH solve rate and wall-clock including
  resets. The reasoning is recorded in a comment in `rover_bare/model.sdf`.

## Known pre-existing, not mine
4 failures in `test_harness_plugin.py` / `test_harness_vec_env.py`. Verified
pre-existing by reverse-applying this session's changes and reproducing them.
Ordering-dependent (each passes alone). Untouched.


## 2026-08-26 ~09:52 -- PHASE 5 RESULT: 97.5% is a plateau, NOT solved

Phase 5 (first training with spawn_redraw active) did not beat the 97.5%
already published. Seven independent 80-episode verifications today:

| checkpoint | screened (24 ep) | VERIFIED (80 ep) |
|------------|------------------|------------------|
| lr3e-5_s0 ck_120000 | 100% | **97.5%** |
| lr8e-5_s0 ck_40000  | 100% | 97.5% |
| lr5e-5_s0 ck_200000 | 100% | 96.2% |
| lr3e-5_s0 ck_180000 | 100% | 92.5% |
| lr3e-5_s0 ck_100000 | 100% | 92.5% |
| lr8e-5_s0 ck_20000  | 100% | 91.2% |
| lr8e-5_s0 ck_60000  | 100% | 80.0% |

**Every one of those screened 24/24.** Small-sample screens are worthless at
this resolution -- they span 80-97.5% on real evaluation. Do NOT report a
24-episode number, and do not caption anything from one.

A hypothesis that turned out FALSE: that a run screening 100% at many
consecutive snapshots (lr3e-5_s0, seven of them) would verify better than a
volatile one. It verified 92.5-97.5%, the same spread. Neighbour-consistency
does not predict true performance here.

**Read: 97.5% is a real ceiling for this approach**, reproduced across three
learning rates and multiple seeds, not a run-to-run accident. The spawn fix
was still correct (it removed episodes that were unwinnable from step 1), but
the residual ~2.5% is genuine difficulty with a valid starting view.

Running: `results/failure_profile.sh` records per-episode termination steps
for the two best checkpoints, to see whether the remaining failures are early
(still a reset/first-frame issue) or mid-episode (a corner the policy cannot
hold). That determines whether 100% is reachable by training at all, or needs
a task/reward change.

NOT DONE: no 100% run exists, so no "solved" video was made. The GUI recording
pipeline is built and verified (`results/record_gui_video.sh`); the in-process
world publishes scene topics so `gz sim -g` attaches to the real randomized
world (the harness backend has NO track randomization and would show the old
single fixed track).


## 2026-08-26 18:03 -- THE SOLVE METRIC WAS REWARDING NOT-DRIVING

Found by Lucas WATCHING the GUI, not from any number: the rover reverses about
half its steps and barely advances. Confirmed and quantified by adding progress
measurement to dr_eval.py (net/path distance, forward efficiency, reversing
fraction; estimated from commanded wheel velocities since HarnessCore exposes
no world-pose read).

| checkpoint | mode | solved | net | path | reversing |
|------------|------|--------|-----|------|-----------|
| single-track Phase 1 | off | 95.0% | 1.63 m | 4.68 m | 39% |
| single-track eighth-DR | off | 92.5% | 1.28 m | 2.96 m | 38% |
| multitrack published | reset | 97.5% | 0.66 m | 1.08 m | 45% |
| phase5 lr3e-5 120k | reset | 97.5% | 0.69 m | 1.52 m | 44% |
| multitrack yesterday | reset | 2.5% | 0.76 m | 2.07 m | 28% |

**Every improvement in solve rate came with LESS driving.** 4.68 -> 2.96 ->
1.08 m of wheel travel across Phase 1 -> DR tuning -> multitrack. And the
checkpoint that drives most (multitrack yesterday, 2.07 m) is the one that
solves worst (2.5%). The relationship is causal, not coincidental:
`terminated_fn` fires ONLY on line loss and "solved" means surviving 300
steps, so committing to speed risks ending the episode while creeping is
always safe. On the 7-shape task the curvature ahead is unknown, which makes
creeping even more attractive -- that is why multitrack policies shuffle more
than single-track ones.

So "97.5% solved" means "kept the line in frame for 300 steps", NOT "drove the
track". Every solve number in this project's history carries that caveat.

**Do not fix this by tweaking the reward weight alone.** `_lf_reward` already
weights forward 1.0x for exactly this reason and it did not prevent the
behaviour, and a previous action-magnitude penalty attempt collapsed the
policy outright (0/160). The structural fix is to make shuffling FATAL rather
than merely lower-scoring: a stall termination (end the episode if net
progress over the last N steps falls below a threshold), and/or a distance
requirement in the definition of solved. Both change the task and require a
retrain from a pre-multitrack base.

`efficiency` alone is misleading and must be read WITH path: the multitrack
policy shows the HIGHEST efficiency (61%) purely because it barely moves.
Low `path` is the damning number.


## 2026-08-27 -- FORWARD-ONLY SPEED CLAMP + GROUND-TRUTH ODOMETRY

Acting on the 2026-08-26 finding above (solve rate was rewarding not-driving).

**Action space changed. Every pre-2026-08-27 checkpoint is now stale** -- they
load and run, but they were optimized against a mapping that no longer exists.

- `clamped_velocities()` in `agent_spec.py` replaces `proportional_velocities`
  for `line_follower`. Forward-only affine map: `a in [-1,1]` ->
  `v in [0.10, 0.51] m/s` per wheel (2.94 to 15.0 rad/s). No stop, no reverse,
  anywhere in the action space. Shuffling is now unreachable rather than
  merely lower-scoring, which is the structural fix STATUS called for -- the
  reward-weighting and action-penalty routes were both already tried and both
  failed. Helper also supports the symmetric `reverse=True` mode (deadzone +
  `v_min` floor) for any spec that does need to back up.
- `AgentSpec.command_limits` + a clamp in `HarnessCore.apply_actions`, applied
  AFTER the gain multiply. `action_to_commands` bounds what the POLICY asks
  for; `action_gain_randomization` scales that afterwards and could push it
  back out of the executable band. `line_follower` caps both axles at
  29.4 rad/s (1.0 m/s, the caster-pop speed).
- Band numbers come from `wmala2/rover-firmware` (500 RPM profile) and are
  written up in `docs/examples/line_follower.md`, "Speed Range And Motor
  Response". `V_MIN`=0.10 m/s is a TRAINING floor, not a measured stiction
  floor -- the firmware's own hard gate is 0.013 m/s.
- Consequence to watch: tightest reachable turn radius is now ~0.12 m
  (`(wheel_sep/2)*(v_max+v_min)/(v_max-v_min)`), so a sharp corner must be
  driven as an arc, not pivoted through. This is the main risk to the change.

**Ground-truth motion measurement, replacing the commanded-velocity estimate.**
`GAZEBO_GYM_ODOM=1` injects `gz-sim-odometry-publisher-system` per agent
(`<prefix>/odom_<i>`); `training_scripts/verify_forward_motion.py` subscribes
over gz-transport and reports path / net / ground speed / signed forward speed
/ reversing fraction, exiting non-zero below the bar. dr_eval's numbers were
estimated from commanded wheel velocities; these are read from the simulator.

- **rover_bare's body-forward axis is +y, not +x.** Measured: full throttle
  gives mean twist `(0.003, 0.481, 0.0)` m/s. Reading `twist.linear.x` (the
  usual convention) reports ~0.000 m/s while the rover is driving at 0.48 m/s.
  Anything else reading this topic must use `--forward-axis y`.

**Control rate was documented wrong for this environment's whole history.**
It is 20 Hz, not 6 Hz: `frame_skip=5` at `max_step_size=0.01` is 50 ms per
action. The 6 Hz figure came from multiplying the 30 Hz camera period by
`frame_skip`. Odometry header stamps settle it -- 60 actions span exactly
3.0 s of simulated time. So one action is ONE firmware PID tick, not three,
which strengthens rather than weakens the case for encoder observations.
Corrected in `docs/examples/line_follower.md`.

**Running:** fresh 4-config PPO sweep under the clamped mapping, single-track,
200k steps, n_agents=4, seed 0, `--checkpoint-every 25000`. Log
`results/clamp_sweep.log`, output `models/line_follower_clamp/`. Single-track
first on purpose: it is the task this project has actually solved before, so
it is the cleanest read on whether the clamp helps or hurts. Multi-track is
the follow-up, not the screen.


## 2026-08-27 -- SWEEP RESULT UNDER THE CLAMP, AND A THREE-CHECK VALIDATION GATE

`training_scripts/validate_rover.py` (renamed from verify_forward_motion.py)
runs three checks against the simulation that actually ran, and exits non-zero
on any failure. `sweep.py --validate` runs it after every trial.

| Check | Passes when | Source |
|---|---|---|
| `camera` | every agent's `camera_link` is pitched 45 deg down | the built world SDF |
| `forward` | path covered, median forward speed, low reversing share | odometry topics |
| `speed` | inside the executable band, not stalled, not above the caster-pop speed | odometry topics |

Camera pitch is read from the world SDF the server was handed, NOT from
`/world/<name>/pose/info`: that message names links unqualified
(`camera_link`, `base_link`) with no model scoping, so all N agents' links
collapse onto each other in a name-keyed read.

**Sweep: 4 PPO configs x 200k steps, single-track, seed 0, n_agents=4.**
~950 s per trial, all four run concurrently.

| trial | lr | ent_coef | n_steps | batch | final mean_ep_reward | validation |
|---|---|---|---|---|---|---|
| 0 | 3e-4 | 0.01 | 64 | 256 | **111.7** | camera 4/4, forward 4/4, speed 4/4 |
| 1 | 1e-3 | 0.0 | 128 | 256 | 95.8 | camera 4/4, forward 4/4, speed 4/4 |
| 2 | 5e-4 | 0.005 | 128 | 512 | 95.7 | camera 4/4, forward 4/4, speed 4/4 |
| 3 | 3e-4 | 0.0 | 256 | 512 | 98.3 | camera 4/4, forward 4/4, speed 4/4 |

**The clamp did what it was for.** Every agent of every trial: ground speed
0.47-0.51 m/s, median forward speed 0.49-0.51, peak 0.50-0.52 (limit 1.00),
reversing ~2%. Compare the 2026-08-26 numbers: 38-45% reversing, 1.08 m of
travel. Creeping and shuffling are gone, structurally, not by tuning.

**But 200k steps does not learn to steer, and the three checks do not catch
that.** Action statistics over 300 steps x 4 agents:

| trial | mean action | steer \|L-R\| mean | max | episodes ended |
|---|---|---|---|---|
| 0 | [0.958, 1.000] | 0.042 | 1.65 | 16 |
| 1 | [0.977, 0.944] | 0.079 | 2.00 | 18 |
| 2 | [0.976, 0.790] | 0.233 | 2.00 | 17 |
| 3 | [1.000, 0.960] | 0.040 | 0.81 | 18 |

All four saturate at full throttle with almost no differential. ~70 steps per
episode out of 300, i.e. they drive the opening straight at 0.5 m/s and lose
the line at the first corner. A policy doing exactly that passes camera,
forward AND speed. **The three checks are necessary, not sufficient** -- they
prove the rover drives realistically, not that it follows the line. Read them
alongside mean_ep_reward, never instead of it.

200k is under-budget by this project's own history: the original single-track
solve needed 400k and was explicitly a slow-convergence problem at 100k, not a
plateau. Next step is 400k, not a different config.

**Open risk, untested because nothing steered yet:** forward-only actuation
gives a minimum turn radius of ~0.12 m. The track corners are sharp 90 deg.
Whether a 0.12 m arc keeps the line inside a camera that sees 7 cm ahead is
the first thing to check if 400k also stalls around 70 steps. If it cannot,
lowering `_LF_WHEEL_SPEED_MIN` widens the radius at the cost of allowing
slower creeping.

**A metric bug found and fixed here, worth not repeating.** Reset teleports
corrupt the odometry stream: the chassis jumps metres between consecutive
samples, which reads as hundreds of m/s. Differencing straight through it gave
a peak speed of 88 m/s and a mean ground speed above the wheels' own full
scale. The teleport corrupts the *twist* on that sample too, not just the
pose difference -- on one run the median forward speed was 0.505 m/s while
the mean over the same samples was 0.033. Discontinuities above 3 m/s are now
dropped from every accumulator and counted in a `resets` column, and the
forward statistic is a median rather than a mean.

**Also fixed:** the world name is now namespaced per process
(`<spec>_<pid>_<uuid8>`), like the camera topics already were. Reproduced the
collision live: a single-agent env subscribed to `/world/line_follower_inproc/
pose/info` and received four agents, from the sweep running in other
processes. Affects `pose/info`, `dynamic_pose/info`, `stats`, and `gz sim -g`
attaching to a world by name.


## 2026-08-27 -- 400k SWEEP: NO IMPROVEMENT. THE CORNER IS THE WALL.

400k, single-track, seed 0, n_agents=4, `--validate` on. ~1950 s per trial.

| trial | lr | ent_coef | n_steps | batch | 200k | 400k | validation |
|---|---|---|---|---|---|---|---|
| 0 | 3e-4 | 0.01 | 64 | 256 | 111.7 | 101.6 | camera/forward/speed 4/4 |
| 1 | 1e-3 | 0.0 | 128 | 256 | 95.8 | 109.3 | camera/forward/speed 4/4 |
| 2 | 5e-4 | 0.005 | 128 | 512 | 95.7 | **113.4** | camera/forward/speed 4/4 |
| 3 | 3e-4 | 0.0 | 256 | 512 | 98.3 | 110.2 | camera/forward/speed 4/4 |

Doubling the budget bought 1.7 reward points on the best trial, and trial 0
got WORSE. Curves: every trial reaches ~110 by 48k steps and is flat for the
remaining 350k. Peaks 110.1-117.3. **My "200k is under-budget" call was
wrong** -- this is a plateau, not slow convergence. Do not spend more steps.

Mean episode length 62-71 of 300. Trials 1 and 3 have a p90 steering
magnitude of exactly 0.000: they drive perfectly straight 90% of the time.

## WHY: the first corner is un-takeable, at EVERY speed in the band

Tested with a hand-written proportional controller on the TRUE centroid (no
learning involved), so this is a property of the task, not of PPO. Sweeping
base speed x gain, 600 steps, 4 agents:

| base speed | mean ep len | distance before failure |
|---|---|---|
| 0.31 m/s | 80-104 | 1.24-1.61 m |
| 0.22 m/s | 130-136 | 1.43-1.50 m |
| 0.16 m/s | 181-192 | 1.45-1.54 m |
| 0.12 m/s | 274 | 1.64 m (some hit the 300 cap) |
| 0.10 m/s | 291-296 | 1.46 m (5 of 8 hit the cap) |

**Episode length varies 3.6x with speed; distance before failure does not
vary at all (~1.4 m).** The rover fails at the same PLACE, not after the same
time. Spawn is (0,-1) driving along -X; the corner is at x=-1.5, i.e. 1.5 m
away. It dies at the first corner regardless of how fast it approaches it.

Slow configurations "solve" only because the 300-step cap (15 s) arrives
before the corner does: at 0.10 m/s an entire episode covers 1.5 m of a ~10 m
loop. That is the creeping loophole reappearing at the floor of the allowed
band -- the clamp raised the floor, it did not close the loophole.

Mechanism: forward-only actuation gives a minimum turn radius of
(wheel_sep/2)*(v_max+v_min)/(v_max-v_min) = ~0.12 m. Turning 90 degrees at
the maximum yaw rate (v_max-v_min)/wheel_sep = 2.5 rad/s takes ~0.63 s, during
which the rover advances ~0.19 m. The camera sees 0.07 m ahead. The line
leaves the frame mid-swing and `terminated_fn` fires.

**A methodology note worth keeping:** the scripted controller initially did
WORSE than the RL policies (max 64 steps) because I had the steering sign
backwards. The camera is mounted at yaw -1.5708, which mirrors image-x
against the rover's left/right, so `left = base - k*e`, NOT `+`. The tell was
that higher gain made it fail FASTER (29 -> 3 -> 2 steps). Any future
hand-written controller or reward term that keys on centroid error needs this
sign.

## What to try next, in order of expected value

1. **Constrain the MEAN of the two wheels to be forward, instead of each one
   individually.** Per-wheel range goes back to [-v_max, v_max] with
   `mean(v_L, v_R) >= v_min` enforced. Net creeping and shuffling stay
   impossible (that constraint is exactly what forbids them), but a tight
   pivot at a corner becomes reachable again. This is the smallest change
   that addresses the measured cause.
2. **Define solved by distance covered, not steps survived.** 300 steps at
   the floor speed is 1.5 m of a 10 m loop. Until the bar requires ground
   covered, creeping remains a winning strategy at whatever the floor is.
3. Rounded track corners, if 1 and 2 are not enough -- but that changes the
   task, so try it last.

NOT worth trying: more steps, other hyperparameters, other algorithms. The
plateau is geometric and reproduces without any learning at all.


## 2026-08-27 -- BOTH FIXES LANDED. TURN RADIUS WAS NOT THE CAUSE.

**(1) `mean_forward_velocities()`** replaces `clamped_velocities(reverse=False)`
on `line_follower`. Each wheel is commanded over the full symmetric
[-v_max, v_max], then the pair is projected onto `mean(v) >= v_min` by an
equal shift, with any excess over v_max handed to the other wheel so the SUM
is preserved and the mean lands exactly on v_min. Creeping and shuffling stay
unreachable (both need a low mean); a tight pivot comes back. Measured
reachable turn radius: **0.020 m, down from 0.12 m**.

Note a design dead end: parameterizing as (forward, turn) from the same two
clipped actions does NOT work. |turn| <= 1 - |forward|, so the tight-pivot
corner of the space is unreachable and the radius stays at 0.12 m. The
projection formulation is what makes it reachable.

**(2) Distance-based solved bar.** `max_episode_steps` 300 -> 600, and
`dr_eval.py --solved-distance` requires an episode to cover N metres AND last
the cap. It prints `solved` and `steps_only` side by side; the gap between
them is the size of the illusion. The single-track loop is ~10 m, so 8.0 is
most of a lap.

### Turn radius was NOT the binding constraint

Re-ran the hand-written controller under the new map, sweeping base speed
(0.10 to 0.50 m/s) x gain (0.3 to 10):

**Distance before failure is 1.08-1.65 m in EVERY cell.** Episode length
varies 6.6x (47 to 330 steps) and speed varies 5x, and distance does not move.
Identical to the 1.24-1.64 m measured under the old 0.12 m-radius map. A 6x
tighter turn radius bought nothing. My turn-radius hypothesis was WRONG.

Bar (2) immediately earned its keep: the best cell (base 0.0, any gain) lasts
310-330 steps but covers 1.55-1.65 m, i.e. 0.10 m/s, exactly the v_min floor.
Under the old steps-only bar that would have read as near-solved. It is
creeping, and the distance bar says so.

### THE ACTUAL CAUSE: the reward is blind to the corner

Logged the last 16 steps before termination, driving the straight:

| step | bottom-half centroid (used) | dark px | line mean row |
|---|---|---|---|
| 326 | 0.410 | 961 | 8.5 |
| 333 | 0.440 | 481 | 3.5 |
| 338 | 0.484 | 125 | 0.7 |
| 341 | 0.519 | 9 | 0.0 |

The centroid stays at **0.41-0.52, dead centre, the entire way in**. The line
does not exit sideways; it RECEDES and shrinks to nothing straight ahead.

`_lf_reward`'s centering term is `1 - 2*|c - 0.5|` on that same bottom-half
centroid, so **the policy is paid full centering reward right up to the step
it terminates**, with no gradient anywhere pointing toward turning. That is
why PPO plateaus at ~110 by 48k steps and why the scripted controller fails
identically: neither has a signal. Turn authority was never the problem;
there was nothing telling anything to use it.

### The corner IS visible, in the half of the frame that gets discarded

Same run, comparing centroids over different image bands:

| step | bottom half (used) | full frame | top half |
|---|---|---|---|
| 322 | 0.405 | 0.519 | **0.874** |
| 330 | 0.428 | 0.501 | 0.639 |
| 338 | 0.484 | 0.499 | 0.502 |

The top half carries a strong signal (0.874 vs 0.5) a full **20 steps, 1
second, before termination**, and it decays as the corner leaves the field of
view, so the warning is strongest earliest -- the right shape for
anticipation. `_lf_line_centroid` slices `img[h//2:]` and throws it away.

**The full-frame centroid is NOT the fix**: bottom (0.41) and top (0.87)
average to 0.50 and cancel. The lookahead has to be a SEPARATE signal.

### Next, in order

1. **Add a lookahead term to the reward** from the top-half centroid, and/or
   from the dark-pixel count, which decays 961 -> 9 over those same 16 steps
   and is an equally graded warning. This is the smallest change that creates
   the missing gradient.
2. Put the same lookahead in the observation as an explicit channel (it is
   already implicitly in the raw image, so this is about making it easy, not
   about making it available).
3. Only then revisit termination -- `terminated_fn` firing on an empty bottom
   half is what turns "no signal" into "no recovery".

Running: 400k sweep under the new map, `--validate` on, results in
`results/meanfwd_sweep.log`. Worth running despite the controller result,
because a CNN sees the raw frame and could learn the top-half cue the
centroid controller is structurally blind to.


## 2026-08-28 -- REWARD REBUILT ON CLASSICAL LINE FOLLOWING; FIDELITY BUGS FIXED

### The wheels were spheres

`rover_bare`/`rover` wheel collisions were `<sphere radius=0.034>`, i.e. a
POINT contact with no resistance to yaw about it, plus anisotropic friction
(mu 1.2 / mu2 0.8) with no `<fdir1>`, so the two directions were whatever the
contact frame happened to be. The chassis could slide and spin nearly freely.

This is the root of several things previously blamed on other causes:
full throttle produced 0.79-0.86 m/s ground speed from wheels that can only
deliver 0.51, and hard turns flung the rover to 1.5-2.95 m/s. **The
"mean_forward_velocities is unphysical" conclusion from 2026-08-27 was wrong
-- the pivots were unstable because of the spheres, not the action map.**

Now `<cylinder radius=0.035 length=0.0265>`, isotropic mu=1.0. Radius is the
firmware's WHEEL_DIAMETER_M/2; length is the CAD wheel's width recovered from
its inertia tensor. Measured after: ground speed 0.485-0.502 against a 0.50
full scale, peak 0.508. Skid gone.

### Other fidelity fixes
- Speeds are now DEFINED in m/s (the hardware's unit) and derived to rad/s,
  not the reverse. Wheel radius 0.034 -> 0.035 everywhere.
- `action_gain_randomization` draws PER WHEEL, not one scalar for the whole
  robot. A shared gain changes how fast a differential drive goes; only a
  mismatch changes where it ends up, and veer is what breaks a line follower.
  The firmware carries separate TRIM_LEFT/TRIM_RIGHT for exactly this.

### Still-open fidelity gaps (NOT fixed)
1. Control rate 20 Hz vs the rover's 10 Hz command playback and 15 fps camera.
2. No actuation latency (UDP RTT + 10 Hz queue + PID ramp); sim applies
   actions instantly.
3. No camera latency; sim reads a clean render, hardware gets stale JPEG.
4. Sim never JPEG-compresses by default; the real camera always does.

### Reward: three designs, measured

| reward | median ep len | median path | speed |
|---|---|---|---|
| original (centred + unconditional speed) | 62-71 | ~1.3 m | ~0.5 m/s |
| classical ADDITIVE (cross+heading+progress+ahead) | 257 | 1.28 m | 0.10 m/s |
| speed-MULTIPLICATIVE (shipped) | 74 | **1.66 m** | 0.45 m/s |

**The additive version re-created the Ant alive-bonus trap.** Its three
always-on positive terms paid 1.75/step regardless of motion, so creeping the
full 600-step episode was worth ~1050 while driving hard and losing the corner
was worth ~165. Episodes got 4x longer while distance did not move: 257 steps
covering 1.28 m is 0.10 m/s, exactly v_min. Its headline mean_ep_reward (389.8
vs the old 113.4) looked like a 3.4x win and was an artifact of reward scale.
**Do not compare mean_ep_reward across a reward change. Use episode length and
path.**

Shipped version is multiplicative in speed:
`r = (v/v_max) * (0.6*q_cross + 0.4*q_heading)`, standing still scores ~0.
Kendall et al., "Learning to Drive in a Day" (arXiv:1807.00412) use forward
speed alone with termination on infraction, for the same reason. Creeping now
pays ~119/episode against ~396 for a completed lap.

`q_heading` pays ONLY when the furthest scan band sees track. Without that
gate it defaults to "perfectly aligned" the moment the lookahead empties --
which is exactly the corner approach -- and a corner frame scored 0.99, same
as an open straight.

### Features: 4 horizontal scan bands, not one blob
`_lf_track_features()` returns cross-track, heading, curvature and
lookahead-visibility from 4 bands. One row gives position only; several give
position AND angle, which is what lets a tracker see a corner coming. Sources:
Stanley (cross-track + heading), Regulated Pure Pursuit (arXiv:2305.20026,
curvature-regulated speed), RoboCup Junior Rescue Line practice for 90 degree
corners ("slow both motors"; "change the speed of the stopped motor to go
backwards" -- note the forward-only clamp FORBIDS both of those tactics).

### Where it stands
Best checkpoint drives 1.66 m at 0.45 m/s, 100% forward efficiency, 0%
reversing, and gets AROUND the first corner (at 1.5 m) before failing shortly
after. 0/16 on the 8 m lap bar. All trials pass camera/forward/speed.

### Running next (results/next_experiments.sh)
PPO paper Table 5 settings this project has never used: LR+clip annealing to
zero, then epochs 3 for vision (our grid uses 6-10, from the CONTINUOUS
CONTROL table). The documented peak-then-collapse pattern across this
project's whole history is the canonical symptom annealing addresses.

### Two process lessons
- A hung `phase5_lr5e-5_s0` trial ran for 2 days holding ~10% RAM and a camera
  slot, contending with every measurement. Killed (ignored SIGTERM). Check for
  stale detached runs before trusting a number.
- Do not wait on `pgrep -f "<long command>"` from a script: the launcher
  process carries the script text on its own command line and pgrep matches
  ITSELF, so the loop never exits. Cost 90 minutes of idle machine.


## 2026-08-28 (later) -- ANNEALING HELPS; THE POLICY NEVER LEARNED TO STEER

PPO paper Table 5 settings, isolated one at a time, all on the multiplicative
reward. Same reward across all three, so mean_ep_reward IS comparable here.

| run | best reward | median ep len | median path |
|---|---|---|---|
| speed-reward baseline | 47.1 | 74 | 1.66 m |
| + LR/clip annealing | 50.7 | 84 | 1.74 m |
| + annealing + 3 epochs | 50.1 | 75 | **1.85 m** |

Annealing: +7.6% reward, +5% distance. 3 epochs: another +6% distance AND 5%
less wall-clock (1840s vs 1940s per trial), so it is strictly better -- same
result for less compute, exactly what the paper's vision table implies. Both
are now the default for this task.

**The more useful signal is the spread.** Four configs spanning lr 3e-4 to
1e-3, ent_coef 0.0 to 0.01, n_steps 64 to 256 all land within ~1.5 reward
points of each other. Hyperparameters have stopped mattering, which means the
binding constraint is elsewhere. Do not spend more sweeps on this grid, and
do NOT bother testing ent_coef separately -- the grid already spans it.

### Root cause: it does not steer at all

Traced the best checkpoint's last 14 steps. It commands [+1.00, +1.00] --
full throttle, both wheels identical, ZERO differential -- every step, while
cross-track error blows out from +0.20 to -0.98 and the near band loses the
line:

    step 64  cross +0.20  action [+1.00,+1.00]
    step 69  cross -0.43  action [+1.00,+1.00]
    step 74  cross -0.98  action [+1.00,+1.00]

Ten steps of visibly degrading error and it never turned a wheel. The distance
gains from annealing came from driving the straight better, not from cornering.

Two candidate causes, one experiment each (results/corner_experiments.sh):

1. **The data is ~81% straight.** Every episode spawns at the same fixed
   mid-straight point (0,-1); 1.5 m of the 1.85 m covered is a straight where
   going straight IS optimal. The policy sees a corner for maybe 3-5 steps per
   episode. It is converging correctly on the data it is given.
   -> `--track-shapes reset` draws a fresh shape and tangent-aligned spawn
   every reset, so corners arrive immediately.
2. **Steering is forbidden and expensive.** The shipped map caps the turn
   radius at 0.12 m and cannot reverse a wheel; steering also costs 40% of
   speed instantly (mean drops 0.5 -> 0.3 m/s), against a reward multiplicative
   in speed.
   -> `line_follower_pivot`, a REGISTERED A/B variant using
   mean_forward_velocities: 0.02 m radius, one wheel may reverse, creeping
   still unreachable.

### Honest comparison against this project's historical numbers

| checkpoint | solved | net fwd | path | reversing | efficiency |
|---|---|---|---|---|---|
| single-track Phase 1 | 95.0% | 1.63 m | 4.68 m | 39% | 35% |
| single-track eighth-DR | 92.5% | 1.28 m | 2.96 m | 38% | 43% |
| multitrack published | 97.5% | 0.66 m | 1.08 m | 45% | 61% |
| today (anneal+ep3) | 0% | **1.85 m** | 1.85 m | **0%** | **100%** |

The 97.5% policy netted 0.66 m of forward progress on a ~10 m loop. It never
drove the track; it survived 300 steps by shuffling near spawn. Today's policy
nets more forward progress than any checkpoint in that table, with zero
reversing -- and scores 0% because the bar now requires 8 m of ground covered.

Three things moved at once and only one is a real regression:
- the BAR changed (300 steps in-frame -> 600 steps AND 8 m);
- the EXPLOIT was removed (creeping/reversing was how the old score was won);
- **episode survival genuinely regressed, 300 -> 75 steps.** Old policies
  loitered indefinitely; this one commits to 0.45 m/s and drives off at the
  corner. The forward-only clamp removed the two tactics RoboCup names for a
  90 degree corner (slow both motors; reverse the inner wheel) without
  supplying a replacement. That is what `line_follower_pivot` is testing.

**Nothing in this project has ever completed a lap**, including every
checkpoint in the table above.
