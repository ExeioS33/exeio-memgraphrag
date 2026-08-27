# MemGraphRAG developer entry points.
#
# Each target runs the same command as the matching CI job
# (.github/workflows/tests.yml, .github/workflows/lint.yml), so a green `make`
# locally means a green pipeline instead of a second, subtly different one.
#
# Run `make install` once first: the other targets use `uv run --no-sync` on
# purpose, because a bare `uv run` re-syncs the environment down to the default
# dependency set and would silently uninstall the api/client extras the suite
# imports.

UV ?= uv
RUN := $(UV) run --no-sync
EXTRAS := --extra api --extra pytest --extra client
PYTEST_ARGS ?=
# Floor, not target: matches --cov-fail-under in the Coverage job.
COVERAGE_FLOOR ?= 50

.DEFAULT_GOAL := help
.PHONY: help install hooks test test-integration lint fmt cov run clean

help:  ## List the available targets
	@grep -hE '^[a-z][a-z-]*:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  %-18s %s\n", $$1, $$2}'

install:  ## Sync the locked dev environment (api + pytest + client extras)
	$(UV) sync --frozen $(EXTRAS)

hooks:  ## Install the pre-commit hooks (one-time)
	$(UV) tool install --force pre-commit
	pre-commit install

test:  ## Run the full test suite, exactly as CI does
	./scripts/test.sh tests -q --tb=short $(PYTEST_ARGS)

test-integration:  ## Also run the tests that dial Postgres/Neo4j (compose must be up)
	./scripts/test.sh tests --run-integration --tb=short $(PYTEST_ARGS)

lint:  ## Check style without changing anything (ruff check + format --check)
	$(RUN) ruff check .
	$(RUN) ruff format --check .

fmt:  ## Apply the fixes `make lint` asks for
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

cov:  ## Measure coverage and enforce the floor; writes htmlcov/
	$(RUN) --with pytest-cov python -m pytest tests \
		--cov=memgraphrag \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=$(COVERAGE_FLOOR) $(PYTEST_ARGS)

run:  ## Start the API server against the local .env
	$(RUN) memgraphrag-server

clean:  ## Drop caches and coverage artefacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find memgraphrag tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +
