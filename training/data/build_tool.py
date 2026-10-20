#!/usr/bin/env python3
from build_common import build_arg_parser, run_builder


def main() -> None:
    parser = build_arg_parser(default_style="tool")
    parser.set_defaults(stats_name="sampling_stats.json")
    args = parser.parse_args()
    run_builder(style="tool", args=args)


if __name__ == "__main__":
    main()
