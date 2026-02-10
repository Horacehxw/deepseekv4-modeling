#!/bin/bash
# Claude Code hook: run tests before git commit/push

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

# Only intercept git commit or git push commands
if ! echo "$COMMAND" | grep -qE '(git commit|git push)'; then
  exit 0
fi

# Run the test suite
OUTPUT=$(python -m pytest test/ -q 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "$OUTPUT" >&2
  echo "" >&2
  echo "BLOCKED: Tests failed. Fix the failures above before committing/pushing." >&2
  exit 2
fi

echo "All tests passed. Proceeding with $COMMAND."
exit 0
