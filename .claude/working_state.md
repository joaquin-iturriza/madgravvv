# Working state — re-injected after every compaction

Not a log. The lab book in `docs/results.tex` is the log. This holds only what would
otherwise have to be rediscovered: what is in flight, what was decided and why, and what
has already been tried and failed. Overwrite freely; nothing here is history.

`precompact_notes_guard.sh` refuses to compact while this file is older than the last
commit.

## In flight

Nothing running. The improvement programme has NOT started and is not authorised.

## Where the project actually is

Measurement chain complete and reviewed end to end. Baseline efficiency at FAR 100/yr is
0.196 ± 0.045 over three seeds; 0.644 with the likelihood ratio. No change from the
improvement plan has been tested. Numbers in `docs/results.tex`, standing facts in
`CLAUDE.md`.

## Decisions that overturned an earlier conclusion

- The shipped LR coefficients (`data/o3a_frozen_lr_off200.npz`) were fitted on the same
  O3a background we measure against. Never use them on our data; `fit_lr.py` refits with
  fold discipline. Worth about six efficiency points.
- The baseline is our own retrained model, not the vendored weights — their training data
  is unreleased so the comparison cannot be made fair. But their number on OUR evaluation
  is measured: 0.788 against our 0.644, and the gap is plausibly training-set size. We
  train on 20k tiles; HPO_TRAIN holds roughly 400k distinct windows at 1 s stride.
- HPO belongs to every experiment, not to one phase. Testing a change at the baseline's
  hyperparameters is conclusive only if the change WINS there.
- Batch size is not a throughput lever on this model — measured flat from 64 to 512.
  AMP fp16 + channels_last is 1.9x. Table below.

## Measured operating point (2026-09-08, V100-32GB, 1x256x128 tile)

| config | samples/s | peak GiB @512 |
|---|---|---|
| fp32 | 1466 | 27.3 |
| fp32 + channels_last | 1322 | 29.0 |
| AMP fp16 | 2006 | 21.0 |
| **AMP fp16 + channels_last** | **2769** | **21.6** |

Throughput flat in batch size (1441 at 64, 1466 at 512): arithmetic-bound already, so a
larger batch buys memory pressure and nothing else. OOM above 512. channels_last ALONE is
slower; it only pays once fp16 tensor cores are engaged. Not yet wired into training.

## Failed, do not retry

- `Planck18.z_at_value` does not exist — module function, and vectorised. Cost: a
  committed script that had never been run.
- `np.gradient` over sorted SAMPLE redshifts hits duplicate abscissae and returns NaN.
  Use a monotone grid and interpolate onto it.
- Deriving the coherence band width from `--window-seconds`; it is fixed by
  `COHERENCE_WINDOW_S`. Ask `band_coefficients` for it.
- The +/- lag ladder wraps onto itself: distinct pairings are n/s - 1, not twice that.
- Two threshold conventions (`searchsorted` against `threshold_at_far`) gave different
  efficiencies for the same nominal FAR. One convention only: `eval/far.py`.
- Guards written and never verified to fire — the CAE-checkpoint check was dead for a
  whole commit because `far_lr.py` never forwarded the key. Check a new guard triggers.
- Heredocs from the Bash tool land in the shell's own cwd, not the project. Use Write or
  an absolute path.

## Next concrete action

Awaiting authorisation. First step when it comes: the saturation study — training-set
size and horizon against efficiency at fixed FAR, aiming to reach or beat 0.788 before
any improvement experiment runs.
