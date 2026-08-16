#!/bin/bash
# Blocks backgrounded gradle/build invocations. Background-task completion
# notifications never reach subagents in this environment, so an agent that
# backgrounds a build waits forever on a signal that cannot arrive. Forcing
# foreground (with a long timeout) is the only reliable pattern here.
input=$(cat)
command=$(printf '%s' "$input" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))' 2>/dev/null)
background=$(printf '%s' "$input" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("run_in_background",False))' 2>/dev/null)

if [ "$background" = "True" ] || [ "$background" = "true" ]; then
  case "$command" in
    *gradlew*|*"gradle "*|*pytest*|*"npm test"*|*"npm run"*|*vitest*)
      echo "BLOCKED by project hook: never run builds/tests in the background here — background completion notifications NEVER fire in this environment, so you would wait forever. Re-run this exact command in the FOREGROUND with timeout 600000 (builds can take 10-20 min on this machine). Do not wait on Monitor/background notifications; poll files with foreground Bash if needed." >&2
      exit 2
      ;;
  esac
fi
exit 0
