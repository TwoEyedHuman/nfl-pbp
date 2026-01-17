# Variables
PYTHON = pipenv run python
STREAMLIT = pipenv run streamlit run
DASHBOARD = dashboard.py
MODEL_DIR = models

# Default target
.PHONY: all
all: help

## help: Show this help message
.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^##' Makefile | sed -e 's/## //g' -e 's/: /:	/g'

## install: Install dependencies using pipenv
.PHONY: install
install:
	pipenv install

## train-lr: Train the Logistic Regression model
.PHONY: train-lr
train-lr:
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) training/logistic_regression.py

## train-gb: Train the Gradient Boosting model
.PHONY: train-gb
train-gb:
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) training/gradient_boosting.py

## train-all: Train all available models
.PHONY: train-all
train-all: train-lr train-gb

## run: Start the Streamlit dashboard
.PHONY: run
run:
	$(STREAMLIT) $(DASHBOARD)

## clean: Remove cached python files and model files
.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f $(MODEL_DIR)/*.pkl

## shell: Enter the pipenv shell
.PHONY: shell
shell:
	pipenv shell