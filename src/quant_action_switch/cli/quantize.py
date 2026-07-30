from ._common import draft_only, parser
def main() -> int:
    parser("quantize", "Reserved quantization entrypoint; always disabled in this draft.").parse_args()
    return draft_only("quantize")
if __name__ == "__main__": raise SystemExit(main())
