.PHONY: help install test clean train evaluate format lint

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies and package"
	@echo "  make test       - Run tests with pytest"
	@echo "  make train      - Train model with default config"
	@echo "  make evaluate   - Evaluate trained model"
	@echo "  make clean      - Remove build artifacts and cache"
	@echo "  make format     - Format code with black and isort"
	@echo "  make lint       - Run linting checks"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

train:
	python train.py --config configs/default_config.yaml

evaluate:
	python evaluate.py --checkpoint models/checkpoints/best_model.pth --mode evaluate

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

format:
	black src/ tests/ train.py evaluate.py
	isort src/ tests/ train.py evaluate.py

lint:
	flake8 src/ tests/ train.py evaluate.py --max-line-length=100 --ignore=E203,W503
	black --check src/ tests/ train.py evaluate.py
