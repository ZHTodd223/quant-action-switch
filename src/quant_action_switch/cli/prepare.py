from ._common import draft_only, parser
def main() -> int:
    parser("prepare", "Validate preparation metadata only.").parse_args()
    return draft_only("prepare")
if __name__ == "__main__": raise SystemExit(main())
