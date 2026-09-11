"""Thin CLI dispatcher: `python -m src.cli <command> <subcommand> ...`.

This repo's convention is to invoke pipeline stages directly as modules
(`python -m src.data.validate ...`, see README.md). This dispatcher does not
replace that -- it just gives the HDFS experiment the command shape described
in docs/HDFS_ANOMALY_DETECTION.md by forwarding argv to the right module, one
`main()` call each. No new abstraction, no argument re-parsing beyond routing.
"""

from __future__ import annotations

import sys


def _dispatch(module_name: str, argv: list[str]) -> int:
    import importlib

    old_argv = sys.argv
    sys.argv = [module_name] + argv
    try:
        module = importlib.import_module(module_name)
        return module.main()
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1

    command = argv.pop(0)

    if command == "dataset":
        if not argv:
            raise SystemExit("usage: dataset <prepare|validate> --name hdfs [...]")
        action = argv.pop(0)
        if "--name" in argv:
            idx = argv.index("--name")
            argv.pop(idx)
            argv.pop(idx)  # drop the "hdfs" value; only one dataset is wired up
        if action == "prepare":
            return _dispatch("src.data.hdfs.build_dataset", argv)
        if action == "download":
            return _dispatch("src.data.hdfs.download", argv)
        if action == "validate":
            return _dispatch("src.data.validate", argv)
        raise SystemExit(f"unknown dataset action: {action}")

    if command == "train":
        return _dispatch("src.training.hdfs_pipeline", argv)

    if command == "merge":
        return _dispatch("src.training.merge_adapter", argv)

    if command == "evaluate":
        return _dispatch("src.evaluation.hdfs_evaluate", argv)

    if command == "compare":
        return _dispatch("src.evaluation.hdfs_compare", argv)

    if command == "infer":
        return _dispatch("src.inference.hdfs_infer", argv)

    if command == "preflight":
        return _dispatch("src.training.hdfs_preflight", argv)

    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
