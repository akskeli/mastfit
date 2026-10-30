#!/usr/bin/env python3
"""Render the pipeline flowchart (Fig. 2) from assets/pipeline_flowchart.dot.

The diagram is defined in **Graphviz**, not in Python:

    assets/pipeline_flowchart.dot     <- edit this

That file is the source of truth. This script is only a convenience wrapper around
`dot`, so there is nothing to keep in sync between a Python layout and the real one.

Editing in VS Code
------------------
    1. install the extension  "Graphviz Interactive Preview"  (tintinweb)
       (optionally also  "Graphviz (dot) language support"  for highlighting)
    2. open  assets/pipeline_flowchart.dot
    3. Cmd/Ctrl+Shift+P  ->  "Graphviz Interactive: Preview Graphviz / Dot"

You get a live preview beside the source: tweak a colour or a label and it re-renders
as you type. That loop is much faster than round-tripping through this script.

Usage
-----
    python scripts/11_pipeline_flowchart.py                  # -> figures/pipeline_flowchart.png
    python scripts/11_pipeline_flowchart.py --format pdf     # vector, for the camera-ready
    python scripts/11_pipeline_flowchart.py --dpi 300
    python scripts/11_pipeline_flowchart.py --src assets/pipeline_flowchart_wide.dot --dpi 600
                                                             # one row per stage, fits one text column

Requires Graphviz (the `dot` binary):

    macOS          brew install graphviz
    Debian/Ubuntu  sudo apt install graphviz
    Windows        winget install graphviz    (or https://graphviz.org/download/)
    conda          conda install -c conda-forge graphviz
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOT_SRC = os.path.join(HERE, os.pardir, "assets", "pipeline_flowchart.dot")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--src", default=DOT_SRC, help="Graphviz source to render")
    args = ap.parse_args()

    if shutil.which("dot") is None:
        sys.exit(
            "ERROR: Graphviz not found (`dot` is not on PATH).\n"
            "  macOS          brew install graphviz\n"
            "  Debian/Ubuntu  sudo apt install graphviz\n"
            "  Windows        winget install graphviz\n"
            "  conda          conda install -c conda-forge graphviz\n"
            "\nThe diagram source is still readable/editable at:\n"
            f"  {os.path.relpath(args.src)}"
        )
    if not os.path.exists(args.src):
        sys.exit(f"ERROR: missing Graphviz source {args.src}")

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.src))[0]      # pipeline_flowchart[_wide]
    out = os.path.join(args.outdir, f"{stem}.{args.format}")

    cmd = ["dot", f"-T{args.format}", args.src, "-o", out]
    if args.format == "png":
        cmd.insert(2, f"-Gdpi={args.dpi}")
    subprocess.run(cmd, check=True)
    print(f"wrote {out}")
    print(f"source: {os.path.relpath(args.src)}  (edit this; live-preview it in VS Code)")


if __name__ == "__main__":
    main()
