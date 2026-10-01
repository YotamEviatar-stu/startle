#!/usr/bin/env bash
# Report version/date of every CARACAS-related clone and whether upstream has moved.
# Read-only except `git fetch`. Usage: bash .claude/skills/cardiac-ica/scripts/check_upstream.sh

HR="$(cd "$(dirname "$0")/../../../.." && pwd)"
REPOS=(
  "$HR/SASICA"
  "$HR/Cardiac_IC_labelling"
  "$HOME/code/tools/SASICA"
  "$HOME/code/tools/SASICA/CARACAS/heart_functions"
  "$HOME/code/tools/heart_functions"
)

for d in "${REPOS[@]}"; do
  echo "## ${d/#$HOME/~}"
  if [ ! -e "$d/.git" ]; then echo "  (not a git checkout — submodule not initialized?)"; continue; fi
  git -C "$d" fetch --all -q 2>/dev/null || echo "  (fetch failed — offline?)"
  echo "  HEAD:     $(git -C "$d" log -1 --format='%h %ad %s' --date=short)"
  up=$(git -C "$d" rev-parse --abbrev-ref '@{u}' 2>/dev/null || echo origin/HEAD)
  echo "  upstream: $up $(git -C "$d" log -1 --format='%h %ad %s' --date=short "$up" 2>/dev/null)"
  echo "  behind:   $(git -C "$d" rev-list --count "HEAD..$up" 2>/dev/null)   ahead: $(git -C "$d" rev-list --count "$up..HEAD" 2>/dev/null)"
  if [ -n "$(git -C "$d" rev-list "HEAD..$up" 2>/dev/null)" ]; then
    echo "  new upstream commits:"
    git -C "$d" log --format='    %h %ad %s' --date=short "HEAD..$up" | head -10
    echo "  files touched:"
    git -C "$d" diff --stat "HEAD" "$up" | tail -8 | sed 's/^/    /'
  fi
  git -C "$d" submodule status 2>/dev/null | sed 's/^/  submodule: /'
done

echo "## CARACAS thresholds in each SASICA.m"
for s in "$HR/SASICA/SASICA.m" "$HOME/code/tools/SASICA/SASICA.m"; do
  echo "  ${s/#$HOME/~}:"; grep -E '^def\.CARACAS\.(thresh|prctl)' "$s" | sed 's/^/    /'
done
