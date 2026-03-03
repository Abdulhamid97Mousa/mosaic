# MOSAIC Makefile
# Common development commands

.PHONY: help install test test-cov test-quick test-unit test-parallel lint format build clean package

# Default target
help:
	@echo "MOSAIC Development Commands"
	@echo "==========================="
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install package in editable mode"
	@echo "  make install-dev    Install with dev dependencies"
	@echo "  make install-chat   Install with chat support"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make test-quick     Run quick tests (skip slow tests)"
	@echo "  make test-unit      Run unit tests only (skip integration)"
	@echo "  make test-parallel  Run tests in parallel with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run all linters (ruff + pyright)"
	@echo "  make format         Format code with ruff"
	@echo "  make typecheck      Run type checking with pyright"
	@echo ""
	@echo "Building:"
	@echo "  make build          Build distribution packages"
	@echo "  make package        Build standalone executable (PyInstaller)"
	@echo "  make clean          Clean build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-chat:
	pip install -e ".[chat]"

# Testing
test:
	pytest gym_gui/tests -v

test-cov:
	pytest gym_gui/tests -v \
		--cov=gym_gui \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-report=html

test-quick:
	pytest gym_gui/tests -v -m "not slow" --tb=short

test-unit:
	pytest gym_gui/tests -v -m "not slow and not integration" --tb=short

test-parallel:
	pytest gym_gui/tests -v \
		-n auto --dist=loadscope \
		--cov=gym_gui \
		--cov-report=term-missing \
		--cov-report=xml

# Code Quality
lint:
	ruff check gym_gui
	pyright gym_gui

format:
	ruff format gym_gui
	ruff check --fix gym_gui

typecheck:
	pyright gym_gui

# Building
build:
	python -m build --no-isolation

build-wheel:
	python -m build --wheel --no-isolation

build-sdist:
	python -m build --sdist --no-isolation

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .coverage.*
	rm -rf htmlcov/
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Packaging (standalone executable)
package:
	pip install -e ".[packaging]"
	pyinstaller mosaic.spec --noconfirm

# Documentation
docs-build:
	cd docs && make html

docs-clean:
	cd docs && make clean
