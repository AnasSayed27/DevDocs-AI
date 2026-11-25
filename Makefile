LOCK_TAG=lock-slice
LOCKED_PATHS=src/schemas.py tests/
DEP_FILES=requirements.txt

lock:
	@test -n "$(MSG)" || { echo "Usage: make lock MSG='lock: slice 1 schemas'"; exit 1; }
	git add $(LOCKED_PATHS)
	git commit -m "$(MSG)"
	git tag -f $(LOCK_TAG)

check:
	@python -m pytest
	@echo "Checking for modified locks..."
	@git diff --exit-code $(LOCK_TAG) -- $(LOCKED_PATHS) >/dev/null || { echo "FAIL: locked files modified"; exit 1; }
	@echo "PASS: Checks passed."

close:
	@test -n "$(MSG)" || { echo "Usage: make close MSG='feat: slice'"; exit 1; }
	git add .
	git commit -m "$(MSG)"
	@git tag -d $(LOCK_TAG) >/dev/null 2>&1 || true

reset-dry:
	git status --short
	git clean -nd

reset:
	git reset --hard $(LOCK_TAG)
	git clean -fd