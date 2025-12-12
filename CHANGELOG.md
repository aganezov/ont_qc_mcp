# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Structured logging across CLI wrappers, utils, tools, and MCP server
- Input validation with size limits and safer fallbacks
- Structured error payloads and optional verbose provenance
- Test expansion for plotting, utils, edge cases, concurrency, and protocol smoke checks

### Changed
- Bounded subprocess capture and safer streaming pipelines
- Parser semantics clarified for missing vs empty histogram blocks

### Fixed
- Sanitized tool output example artifacts to remove machine-specific paths

