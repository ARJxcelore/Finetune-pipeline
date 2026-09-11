from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

TEMPLATE = Path("configs/train.template.yml")

# Discovered from the template itself (${VAR} placeholders) rather than a
# hardcoded list -- different templates (generic vs HDFS) reference different
# variables, and a static list here would need to stay in sync with every
# template by hand, which is exactly how a new template variable silently
# fails to get substituted.
PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=TEMPLATE, help="Template to render (default: the generic pipeline template)")
    args = parser.parse_args()

    template = args.template.read_text(encoding="utf-8")
    required = sorted(set(PLACEHOLDER_RE.findall(template)))
    missing = [key for key in required if key not in os.environ]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")

    rendered = template
    for key in required:
        rendered = rendered.replace("${" + key + "}", os.environ[key])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Rendered Axolotl config: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
