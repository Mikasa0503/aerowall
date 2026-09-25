# AeroWall Release Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare the selected C350-based AeroWall project as a reproducible, clearly scoped source release without publishing runtime caches or overstating experimental results.

**Architecture:** Keep project code and stable research entrypoints in the repository; add canonical project, contribution, reproducibility, model, and upstream-license documentation. Ship only the accepted demo and the two small baseline checkpoints; exclude simulator installations, scratch worktrees, raw rollouts, logs, and intermediate weights.

**Tech Stack:** Python 3.10, Isaac Sim 2023.1.0-hotfix.1, pinned VolleyBots/JuggleRL/HCSP references, Markdown, JSON, TOML, and Git.

---

### Task 1: Freeze release scope and evidence

- Record C350 as the selected baseline: frozen Launch followed by the C350 rally actor.
- Distinguish the 128-seed-batch capped result from the single-seed 25-rally presentation result.
- Summarize extension results and unfinished work using the user-provided status.
- Preserve pre-existing user changes and experimental history.

### Task 2: Add standard project metadata and license

- Add the selected MIT license for AeroWall-owned code.
- Add `pyproject.toml` metadata and Python/tool constraints without claiming Isaac Sim is a pip dependency.
- Document separately licensed upstream components and the NVIDIA runtime boundary.

### Task 3: Curate release assets and evidence

- Publish the accepted demo MP4 and the original `launch-wall.pt` and `c-u350.pt` filenames.
- Add a model card and compact, path-sanitized C350 evaluation record with source-report and checkpoint hashes.
- Keep scratch runtime, raw reports, logs, generated videos, and non-selected checkpoints outside the Git release.

### Task 4: Standardize project entry documentation

- Rewrite the root README as the release landing page with the research contribution, selected result, limitations, setup, and documentation index.
- Add concise contribution/status and reproduction pages consistent with the project configuration.
- Keep historical experiment notes available as historical evidence rather than the main project description.

### Task 5: Define repository hygiene and verify release boundary

- Extend `.gitignore` for nested scratch work directories and simulator outputs while explicitly retaining the two selected checkpoint files.
- Check that all linked release assets exist, verify their SHA-256 values, parse TOML/JSON, and inspect the final diff/status.
- Do not run simulator experiments or the test suite as part of documentation and packaging work.
