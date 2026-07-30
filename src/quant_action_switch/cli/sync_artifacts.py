from ._common import draft_only, parser
def main() -> int:
    parser("sync", "Reserved artifact-sync entrypoint; no transfer is performed.").parse_args()
    return draft_only("sync")
if __name__ == "__main__": raise SystemExit(main())
