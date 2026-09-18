# Pilot-01 is a development run, not the final result

The original outputs are retained for provenance. This run exposed duplicate weight
cleanup in the experiment wrapper (ignored OptLM destructor exceptions). Cleanup is
now idempotent only in the experiment subclass; original generation code is unchanged.

The initial E3 scheduler did not backfill H2D gaps before reserved future activation
transfers. This can disadvantage the contiguous baseline's second-buffer prefetch.
The corrected scheduler and a regression test are used by smoke-01. Do not use this
pilot's E3 predictions as research evidence. Its memory peaks may also include a
diagnostic model retained longer than intended. Use smoke-01 and its source snapshot.
