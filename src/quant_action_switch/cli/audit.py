from ._common import draft_only, parser
def main() -> int:
    parser("audit", "Inspect repository controls without model execution.").parse_args()
    return draft_only("audit")
if __name__ == "__main__": raise SystemExit(main())
