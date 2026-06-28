# Repository guide for AI agents

Context and conventions for AI coding/reviewing agents (e.g. Codex, Claude)
working in **ont_qc_mcp** — a Model Context Protocol (MCP) server that wraps
bioinformatics CLI tools (`nanoq`, `chopper`, `cramino`, `mosdepth`, `samtools`,
`bcftools`) and a containerized IGV to produce QC statistics for Oxford Nanopore
sequencing data (FASTQ / BAM / CRAM / VCF / BED).

## Working conventions

- **Branches & commits:** never commit to `main`. Use a feature branch and small
  [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
  `chore:`, `docs:`, `ci:`).
- **Verify before commit:** run `scripts/ci-local.sh` (reproduces the CI gate in a
  fresh venv), or quickly: `ruff check . && mypy ont_qc_mcp tests && pytest`.
  (`ruff format --check` enforcement is added in a later quality-gates step.)
- **Merge gates are deterministic:** `ruff`, `mypy`, `pytest`, CodeQL. AI review
  is **advisory** and never blocks a merge.
- **Untrusted input:** file paths, CLI flags, and BED/region content arrive from
  MCP clients and user files — treat them as untrusted. Stay alert to command/
  argument injection and path traversal.

## Code review output contract (BOTH reviewers follow this)

When reviewing a pull request, post **one structured summary comment** shaped so a
maintainer can triage it in two seconds and expand only what matters:

1. **Verdict line + counts.**
   `## 🤖 <Reviewer> review — <emoji> <disposition>`
   then `> <one-sentence summary>. **N blockers · M suggestions · K nits**`
2. **Findings table** — `| Sev | Location | Finding |`, one row per issue, with
   `file:line` locations.
3. **One collapsed `<details>` per finding** — the `<summary>` is the glanceable
   title (severity + `file:line` + short title); the body holds the explanation,
   why it matters, and a ` ```suggestion ` block when a concrete fix applies.

**Severity taxonomy (shared):** 🔴 Blocker · 🟠 High · 🟡 Medium · 🔵 Nit · 💭 Question.

**Anti-noise rules:**

- Do **not** comment on formatting/style — `ruff format` owns that.
- Do **not** restate the diff or narrate what the PR obviously does.
- Do **not** duplicate findings CodeQL already reports.
- Prioritize correctness, security (untrusted-input handling), and
  maintainability over personal preference.
- Be specific and cite `file:line`. Keep it advisory.

End with a one-line footer naming the reviewer/model (so when both bots comment,
it is clear who said what).
