# Mosdepth threshold percentages

Mosdepth counts how many bases in each BED interval have depth at or above each
requested threshold. Its `thresholds.bed.gz` contains **base counts**. The separate
coverage distribution files contain fractions.

The MCP tool returns percentages, so the conversion is:

`percentage = 100 × covered bases / (end − start)`

The old parser multiplied the count by 100 without dividing by interval length.
For 50 covered bases in a 100-base interval, it returned 5000% instead of 50%.
The existing integration tests checked that percentage fields existed, but did
not check their values.

## Reproducible fixture

The regression test creates a 200-base reference and 20 unpaired primary reads:
ten 50-base reads and ten 25-base reads, all starting at the first base. The depth
is exactly 20 on BED `[0,25)`, 10 on `[25,50)`, and zero on `[50,200)`.

| BED interval | Counts at ≥1x / ≥10x / ≥20x | Correct percentages |
| --- | --- | --- |
| `[0,1)` | 1 / 1 / 1 | 100 / 100 / 100 |
| `[0,50)` | 50 / 50 / 25 | 100 / 100 / 50 |
| `[0,100)` | 50 / 50 / 25 | 50 / 50 / 25 |
| `[25,50)` | 25 / 25 / 0 | 100 / 100 / 0 |
| `[50,100)` | 0 / 0 / 0 | 0 / 0 / 0 |

The test runs real samtools and mosdepth through the MCP server, for both named
and unnamed BED files. It checks exact expected percentages and mean depths.
Run it with both tools on `PATH`:

```sh
uv run pytest -q -m integration tests/test_mosdepth_thresholds.py
```

Parser-only checks, including fractional results and invalid interval lengths:

```sh
uv run pytest -q tests/test_mosdepth_thresholds.py
```

## Version comparison

The base-count interpretation is the same in mosdepth 0.3.12 and 0.3.14; upgrading
alone does not correct the parser. CI now tests 0.3.14, whose upstream release
fixes a separate case where a requested region starts beyond a chromosome's end.

Sources: [mosdepth threshold format](https://github.com/brentp/mosdepth/tree/v0.3.14#thresholds),
[0.3.14 release](https://github.com/brentp/mosdepth/releases/tag/v0.3.14).
