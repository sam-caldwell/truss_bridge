# Development tasks for truss_bridge. Run `make help` for the list of targets.

PYTHON ?= python
PACKAGE := truss_bridge
DIST := dist
ARCHIVE := $(DIST)/$(PACKAGE).zip
# E203 (whitespace before ':') conflicts with the slice formatting used in the code.
FLAKE8_FLAGS := --max-line-length 120 --extend-ignore E203

.DEFAULT_GOAL := help
.PHONY: help clean lint test build

help: ## Show this help text
	@echo "Usage: make <target> [PYTHON=/path/to/python]"
	@echo
	@echo "Targets:"
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{ printf "  %-8s %s\n", $$1, $$2 }'
	@echo
	@echo "PYTHON (currently '$(PYTHON)') must have numpy, scipy, matplotlib and flake8 installed."

clean: ## Remove caches, build output and the CLI's output/ drawings
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(DIST)
	rm -rf output
	rm -f bridge_*.png

lint: ## Run the flake8 linter over the package and tests
	$(PYTHON) -m flake8 $(FLAKE8_FLAGS) $(PACKAGE) tests

test: ## Run every unit and integration test, including the slow performance tests
	$(PYTHON) -m unittest discover -s tests -t .

build: clean ## Package truss_bridge/, README.md and example.csv into dist/truss_bridge.zip
	mkdir -p $(DIST)
	zip -r -q $(ARCHIVE) $(PACKAGE) README.md example.csv -x '*/__pycache__/*' '*.DS_Store' 'output/*' '*/output/*'
	@echo "built $(ARCHIVE)"
