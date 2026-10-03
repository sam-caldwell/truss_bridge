# Development tasks for truss_bridge. Run `make help` for the list of targets.

VENV := .venv
# Python used to create the virtual environment (3.10 or later).
BOOTSTRAP_PYTHON ?= python3
# Use the virtual environment once `make deps` has created it.
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python)
PACKAGE := truss_bridge
DIST := dist
ARCHIVE := $(DIST)/$(PACKAGE).zip
# E203 (whitespace before ':') conflicts with the slice formatting used in the code.
FLAKE8_FLAGS := --max-line-length 120 --extend-ignore E203

.DEFAULT_GOAL := help
.PHONY: help deps clean lint test build

help: ## Show this help text
	@echo "Usage: make <target> [PYTHON=/path/to/python]"
	@echo
	@echo "Targets:"
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{ printf "  %-8s %s\n", $$1, $$2 }'
	@echo
	@echo "PYTHON is currently '$(PYTHON)'. Run 'make deps' to set up $(VENV)/ from requirements.txt;"
	@echo "later targets then use it automatically."

deps: ## Create the virtual environment ./.venv/ and install requirements.txt into it
	$(BOOTSTRAP_PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt
	@echo "installed into $(VENV)/; activate with: source $(VENV)/bin/activate"

clean: ## Remove caches, build output and the CLI's output/ drawings
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(DIST)
	rm -rf output
	rm -f bridge_*.png

lint: ## Run the flake8 linter over the package and tests
	$(PYTHON) -m flake8 $(FLAKE8_FLAGS) $(PACKAGE) tests

test: ## Run every unit and integration test, including the slow performance tests
	$(PYTHON) -m unittest discover -s tests -t .

build: clean ## Package the code, docs, requirements.txt and example.csv into dist/truss_bridge.zip
	mkdir -p $(DIST)
	zip -r -q $(ARCHIVE) $(PACKAGE) README.md LICENSE.txt requirements.txt example.csv -x '*/__pycache__/*' '*.DS_Store' 'output/*' '*/output/*' '$(VENV)/*' '*/$(VENV)/*'
	@echo "built $(ARCHIVE)"
