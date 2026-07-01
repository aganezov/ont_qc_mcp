# Repository guide for AI agents

Conventions for the AI agents (Codex, Claude) and humans working in **ont_qc_mcp** — a
Model Context Protocol (MCP) server that wraps bioinformatics CLIs (`nanoq`, `chopper`,
`cramino`, `mosdepth`, `samtools`, `bcftools`) and a containerized IGV to produce QC
statistics for Oxford Nanopore sequencing data (FASTQ / BAM / CRAM / VCF / BED).

> Committed on purpose — the shared rulebook every contributor reads. Keep it
> **public-safe**: process and conventions only; secrets, credentials, and
> unpatched-vulnerability details never go in tracked files.

**Setup and commands live in [CONTRIBUTING.md](CONTRIBUTING.md).** This guide adds the
conventions specific to how we ship and review changes.

## Contribution workflow

1. **Issue first** for non-trivial work — file a GitHub Issue with the
   [bug template](.github/ISSUE_TEMPLATE/bug_report.md). Point at code with **durable
   permalinks** (`…/blob/<sha>/path#Lx-Ly`), never a bare `file:line`.
2. **Branch** off `main` as `<type>/<issue#>-<slug>`; never commit to `main`.
3. **Tests travel with the change** — new behavior ships with tests; for bug fixes go
   **test-first** (a failing regression test that fails for the *right* reason, then the fix).
4. **Small [Conventional Commits](https://www.conventionalcommits.org/)** —
   `feat:` · `fix:` · `chore:` · `docs:` · `ci:`.
5. **PR with `Closes #N`**; update `CHANGELOG.md` for user-visible changes. Keep the diff
   focused — file adjacent bugs as *new* issues, and don't let an autoformatter pull
   unrelated code into it.

## Review & merge

- **Deterministic gates:** `ruff`, `mypy`, `pytest`, CodeQL. Read pass/fail from the
  authoritative run conclusion / check-runs; diagnose any failure from real logs.
- **Dual review is advisory** — every PR gets a Claude *and* a Codex review; neither
  blocks, but **wait for both on the latest commit** before merging.
- **Codex signals:** a 👍 reaction (often no comment) = no findings; a comment = findings.
  Its review can lag — wait rather than re-trigger.
- **Merge when clean:** 0 blockers, required gates green, mergeable. Never merge over an
  actionable finding — fix it or file a follow-up issue.
- **Verify a review's claims before acting** — reviewers can cite the wrong line,
  fabricate a SHA/API, or misstate impact.

## Attribution

Only **commit messages** carry the `Co-Authored-By:` trailer. PR bodies, issues, and
comments carry **no** trailer or footer.

## Untrusted input & security

File paths, CLI flags, and BED/region content arrive from MCP clients and user files —
treat them as **untrusted**; watch for command/argument injection and path traversal.
When fixing a class of bug (especially security), enumerate the **sink pattern** across
*all* call sites, not one grepped name. Report vulnerabilities per
[SECURITY.md](SECURITY.md) — a **private** advisory, never a public issue.

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
