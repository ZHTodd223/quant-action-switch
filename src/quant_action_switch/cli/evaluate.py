from ._common import draft_only, parser
def main() -> int:
    parser("evaluate", "Reserved evaluation entrypoint; no inference is performed.").parse_args()
    return draft_only("evaluate")
if __name__ == "__main__": raise SystemExit(main())
