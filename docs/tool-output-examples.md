# API v2 wire examples

Representative request, response, and error objects live in
[`tests/fixtures/api_v2/`](../tests/fixtures/api_v2/). These fixtures are
schema-validated contract examples. They are not results computed from the
repository's bioinformatics fixtures and must not be interpreted as scientific
evidence.

The public server returns one JSON text content item. Successful numerical calls
include the resolved grouping, effective normalized request, requested result
sections, and backend provenance. Supporting tools retain their focused response
models. Validation errors use `kind: "validation_error"` with field locations and
codes. Execution errors use `kind: "execution_error"`, name the failed stage and
backend, and always set `partial_result_returned` to false.

To inspect the live schemas and resources through MCP, start the server and use
`list_tools`, `list_resources`, and `read_resource`. To run representative calls
against local inputs, use:

```bash
scripts/with-env.sh python scripts/mcp_smoke_real.py --dir /path/to/test_dir
```

The smoke script calls `environment_status`, `read_qc` for a discovered FASTQ,
and both `alignment_qc` and `coverage_qc` for a discovered BAM or CRAM. Add
`--include-error-profile` to request the optional alignment error section. CRAM
calls require `--reference` with an indexed, uncompressed FASTA.
