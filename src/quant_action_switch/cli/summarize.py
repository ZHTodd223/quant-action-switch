from ._common import draft_only, parser
def main() -> int:
    parser("summarize", "Reserved derived-summary entrypoint.").parse_args()
    return draft_only("summarize")
if __name__ == "__main__": raise SystemExit(main())
