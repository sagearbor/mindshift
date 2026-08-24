# Ported from gauge@2157433 server/nudge_policy.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# Foundation A (2026-08-24): the implementation moved to the flat
# server/nudge_policy.py so the phone's realtime path can share it (the
# watch's WS ingest and the phone's audio pipeline need the SAME "how much
# escalation before we alert" brain). This module is a deliberate thin
# re-export so every existing ``from watch.nudge_policy import NudgePolicy``
# keeps working unchanged — do not add behaviour here; edit the flat module.
from nudge_policy import DEFAULT_CHANNELS, NudgePolicy

__all__ = ["DEFAULT_CHANNELS", "NudgePolicy"]
