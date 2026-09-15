"""Reject incomplete v2 evidence after an otherwise successful real mosdepth run."""

import json
import shutil
import subprocess
import sys
from typing import cast

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools


pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", ["whole", "window"])
@pytest.mark.parametrize("truncate", [False, True], ids=["intact", "missing-positive-contig"])
def test_summary_integrity_through_mcp(tmp_path, mcp_server_params, mode, truncate):
    require_executable_tools(["samtools", "mosdepth"])
    sam = tmp_path / "reads.sam"
    bam = tmp_path / "reads.bam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chrA\tLN:1000\n"
        + "".join(f"r{i}\t0\tchrA\t1\t60\t1000M\t*\t0\t0\t{'A' * 1000}\t{'I' * 1000}\tNM:i:0\n" for i in range(2))
    )
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True)
    subprocess.run(["samtools", "index", str(bam)], check=True)
    real_mosdepth = shutil.which("mosdepth")
    assert real_mosdepth is not None
    shim = tmp_path / "mosdepth-shim"
    receipt = tmp_path / "mosdepth-receipt.json"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import gzip, json, subprocess, sys\n"
        "from pathlib import Path\n"
        f"result = subprocess.run([{real_mosdepth!r}, *sys.argv[1:]])\n"
        "if result.returncode == 0 and '--version' not in sys.argv:\n"
        "    per_base = Path(sys.argv[-2] + '.per-base.bed.gz')\n"
        "    with gzip.open(per_base, 'rt') as stream:\n"
        "        text = stream.read()\n"
        "    rows = text.splitlines()\n"
        f"    if {truncate!r}:\n"
        "        rows = [row for row in rows if not row.startswith('chrA\\t')]\n"
        "        with gzip.open(per_base, 'wt') as stream:\n"
        "            stream.write('\\n'.join(rows) + ('\\n' if rows else ''))\n"
        f"    Path({str(receipt)!r}).write_text(json.dumps({{'returncode': result.returncode, "
        "'args': sys.argv[1:], 'original': text, 'delivered_rows': rows}))\n"
        "sys.exit(result.returncode)\n"
    )
    shim.chmod(0o755)
    params = mcp_server_params.model_copy(update={"env": {**(mcp_server_params.env or {}), "MOSDEPTH": str(shim)}})

    async def check():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                arguments: dict[str, object] = {"path": str(bam)}
                if mode == "window":
                    arguments["window_size"] = 400
                result = await session.call_tool("coverage_qc", arguments)
                proof = json.loads(receipt.read_text())
                assert proof["returncode"] == 0
                assert "--by" in proof["args"]
                assert bool(proof["original"]) is True
                assert (proof["delivered_rows"] == []) == truncate
                content = cast(types.TextContent, result.content[0]).text
                if truncate:
                    assert result.is_error, content
                    assert "missing or inconsistent with positive-depth evidence" in content
                else:
                    assert not result.is_error, content
                    payload = json.loads(content)
                    assert {(row["chrom"], row["mean_depth"]) for row in payload["rows"]} == {("chrA", 2)}

    anyio.run(check)
