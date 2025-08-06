# Monkey-Tail Makefile
# Cross-platform build and development tasks

.PHONY: all build test clean fmt clippy bench install docs help

# Default target
all: fmt clippy test build

# Build all crates
build:
	cargo build --release

# Build with all sensor features
build-full:
	cargo build --release --all-features

# Run all tests
test:
	cargo test --all

# Run tests with all features
test-full:
	cargo test --all --all-features

# Clean build artifacts
clean:
	cargo clean
	rm -rf target/
	rm -rf */target/

# Format code
fmt:
	cargo fmt --all

# Run clippy linter
clippy:
	cargo clippy --all -- -D warnings

# Run clippy with all features
clippy-full:
	cargo clippy --all --all-features -- -D warnings

# Run benchmarks
bench:
	cargo bench

# Install binary globally
install:
	cargo install --path . --force

# Generate documentation
docs:
	cargo doc --all --no-deps --open

# Generate documentation with all features
docs-full:
	cargo doc --all --all-features --no-deps --open

# Development tasks
dev-setup:
	rustup component add rustfmt clippy rust-src
	rustup target add wasm32-unknown-unknown
	cargo install wasm-pack

# Python bindings
python-bindings:
	cd monkey-tail-bindings && maturin develop --features python

# WebAssembly bindings
wasm-bindings:
	cd monkey-tail-bindings && wasm-pack build --target web --features wasm

# Run examples
example-basic:
	cargo run --example basic_trail_extraction

example-sensors:
	cargo run --example multi_modal_sensors --features all-sensors

# Performance profiling
profile:
	cargo build --release
	perf record --call-graph=dwarf target/release/monkey-tail extract --sensors audio
	perf report

# Memory profiling with valgrind
memcheck:
	cargo build
	valgrind --tool=memcheck --leak-check=full target/debug/monkey-tail extract

# Coverage analysis
coverage:
	cargo tarpaulin --all --out Html --output-dir coverage/

# Security audit
audit:
	cargo audit

# Check dependencies for updates
update-deps:
	cargo update

# Quick development check
check: fmt clippy test

# Release preparation
release-prep: clean fmt clippy test-full build-full docs-full

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Format, lint, test, and build"
	@echo "  build         - Build all crates in release mode"
	@echo "  build-full    - Build with all sensor features"
	@echo "  test          - Run all tests"
	@echo "  test-full     - Run tests with all features"
	@echo "  clean         - Clean build artifacts"
	@echo "  fmt           - Format code"
	@echo "  clippy        - Run clippy linter"
	@echo "  clippy-full   - Run clippy with all features"
	@echo "  bench         - Run benchmarks"
	@echo "  install       - Install binary globally"
	@echo "  docs          - Generate documentation"
	@echo "  docs-full     - Generate docs with all features"
	@echo "  dev-setup     - Install development dependencies"
	@echo "  python-bindings - Build Python bindings"
	@echo "  wasm-bindings - Build WebAssembly bindings"
	@echo "  profile       - Run performance profiling"
	@echo "  memcheck      - Run memory leak detection"
	@echo "  coverage      - Generate test coverage report"
	@echo "  audit         - Security audit of dependencies"
	@echo "  check         - Quick development check"
	@echo "  release-prep  - Prepare for release"
	@echo "  help          - Show this help message"