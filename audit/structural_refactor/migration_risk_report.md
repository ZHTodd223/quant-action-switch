# Migration risk report

- Frozen evidence paths are retained and hash-baselined.
- Legacy core paths use compatibility wrappers.
- Only legacy scripts with no current in-repository reference were moved.
- `scripts/_compat.py` is an explicit, temporary sys.path exception.
