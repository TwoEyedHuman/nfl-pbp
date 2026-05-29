# Variables
PYTHON = pipenv run python
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

## freeze: Export dependencies to requirements.txt
.PHONY: freeze
freeze:
	pipenv requirements > requirements.txt

## dev: Start the Streamlit dashboard locally
.PHONY: dev
dev:
	pipenv run streamlit run dashboard.py

## build: Build the Docker image
.PHONY: build
build:
	docker build -t nfl-win-probability .

## run: Run the Docker container
.PHONY: run
run:
	docker run -p 8501:8501 --env CACHE_DIR=/app/cache nfl-win-probability

## down: Stop the Docker container
.PHONY: down
down:
	docker stop $$(docker ps -q --filter ancestor=nfl-win-probability)

## test: Run all unit tests
.PHONY: test
test:
	pipenv run pytest tests/

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

## clean: Remove cached python files and model files
.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f $(MODEL_DIR)/*.pkl

## shell: Enter the pipenv shell
.PHONY: shell
shell:
	pipenv shell
