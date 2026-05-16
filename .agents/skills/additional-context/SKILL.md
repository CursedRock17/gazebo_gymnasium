---
name: additional-context
description: Use git workflow to check other branches and work history for better context
disable-model-invocation: false
allowed-tools: Bash(git add *) Bash(git commit *) Bash(git status *) Bash(git show *) Bash(git worktree add*) Bash(git worktree remove*)
---

## Checkout another branch for original context
- Check Current Branch: !`git branch`
- Check Current Status: !`git status`

- View a file from reference branch without switching
```bash
git show reference:path/to/file.py
```
- Or check it out temporarily into a separate folder
```bash
git worktree add ../reference-draft reference
```

## Your Task
Use additional context from other implementations to see if information is valuable for this branch
