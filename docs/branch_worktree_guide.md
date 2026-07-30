# Branch and worktree guide

Branches describe a line of code or evidence history; a worktree is only a
local checkout of one branch or commit.  Keep those concepts separate so that
an experiment directory is never mistaken for a branch.

## Current branch roles

| Ref family | Role | Handling |
| --- | --- | --- |
| `agent/domestic-cache` | Main integration branch and `origin/HEAD` | Start new repository-maintenance work here after a clean fast-forward. |
| `fix/*` | Narrow, reviewable runtime/control repairs | Merge only after tests and evidence contracts pass; do not use for new result data. |
| `results/*` | Append-only experiment-attempt and result history | Keep separate from implementation changes; do not rebase or force-push. |
| `backup/*` | Preservation point for an earlier integration state | Retain until its replacement and remote availability are verified. |
| detached worktrees | Temporary audit/reproduction snapshots | Treat as read-only unless the owning task explicitly resumes them. |

## Local worktree rules

- One active task per named worktree.  Do not share a worktree between an
  execution attempt and code maintenance.
- Before removal, verify the worktree is clean, the commit is reachable from a
  retained branch or tag, and any experiment evidence is independently backed
  up and hash-verified.
- Never remove a detached audit worktree solely because it is old; first
  identify its owner and whether it contains unmerged evidence.
- Use `git worktree list --porcelain` and `git branch -vv` to refresh this
  inventory.  This document states policy, not a time-sensitive snapshot.

## Suggested naming

Use `fix/<scope>-vN` for code-only repairs, `results/<matrix>-vN` for
evidence-only commits, and `audit/<question>-vN` for a review branch.  Keep
attempt identifiers in `formal_experiments/attempts/`; do not encode local
directory names into branch semantics.
