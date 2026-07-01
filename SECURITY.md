# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities **privately** via GitHub Security Advisories — open a
draft at the repo's **Security → Advisories** tab ([new advisory](../../security/advisories/new)) —
not a public issue or PR. We aim to acknowledge within a few days, agree a fix and
disclosure timeline, and credit reporters who wish to be named.

## Scope

`ont_qc_mcp` wraps external CLIs and runs them on user-supplied inputs (file paths, CLI
flags, BED/region content). Command/argument injection, path traversal, and unsafe
handling of untrusted input are especially in scope.

## Supported versions

Pre-1.0 and unreleased; fixes land on `main`. This section will list supported ranges
once versioned releases exist.
