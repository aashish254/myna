# The published numbers, and the four ways to touch them.
#
#   make repro          check every table row against its committed witness (the gate)
#   make rows           print the registry as a markdown table
#   make repro-show ROW=needle           print one row's command and the files it would
#                       overwrite; run nothing
#   make repro-run ROW=needle            re-run one row FOR REAL: it overwrites that
#                       row's witness, which is why the row name is a variable and
#                       --yes is not the default (SPEC §9.30)
#
# `repro` exits non-zero on the first row whose artifact drifted from the prose, so it
# belongs in CI ahead of any doc change. The runner refuses a `gated-kaggle` row (there
# is no local command for it) and a `retrain` row (training does not happen on this
# box); both refusals are the point, not an inconvenience.

.PHONY: repro rows repro-show repro-run

PY = uv run python

repro:
	$(PY) bench/reproduce.py --check

rows:
	$(PY) bench/reproduce.py --list

define need_row
$(if $(ROW),,$(error usage: make $(1) ROW=<id>   -- see make rows))
endef

repro-show:
	$(call need_row,repro-show)
	$(PY) bench/reproduce.py --run $(ROW) --yes --dry-run

repro-run:
	$(call need_row,repro-run)
	$(PY) bench/reproduce.py --run $(ROW) --yes
