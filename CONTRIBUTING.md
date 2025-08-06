# Contributing to Monkey-Tail

Thank you for your interest in contributing to the Monkey-Tail framework! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

### Prerequisites

- Rust 1.70+ with stable toolchain
- Git for version control
- Basic understanding of signal processing and pattern recognition concepts

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/fullscreen-triangle/monkey-tail.git
cd monkey-tail
```

2. Install development dependencies:
```bash
make dev-setup
```

3. Build the project:
```bash
make build
```

4. Run tests to verify setup:
```bash
make test
```

## Development Workflow

### Branch Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for new features
- `feature/*`: Feature development branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical fixes for production

### Making Changes

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the coding standards below

3. Ensure tests pass:
```bash
make check
```

4. Commit with descriptive messages:
```bash
git commit -m "feat: add thermodynamic pattern persistence detection"
```

5. Push and create a pull request

### Commit Message Format

Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## Coding Standards

### Rust Guidelines

- Follow standard Rust formatting (`cargo fmt`)
- Pass all clippy lints (`cargo clippy`)
- Maintain 80% test coverage minimum
- Use meaningful variable and function names
- Add comprehensive documentation for public APIs

### Documentation

- Use rustdoc comments for all public items
- Include examples in documentation
- Update README.md for significant changes
- Add inline comments for complex algorithms

### Example Documentation:

```rust
/// Extracts thermodynamic trails from multi-modal sensor streams using
/// progressive noise reduction.
/// 
/// # Arguments
/// 
/// * `sensors` - The sensor environment providing data streams
/// * `config` - Configuration parameters for noise reduction
/// 
/// # Returns
/// 
/// A vector of extracted thermodynamic trails with their persistence scores
/// 
/// # Examples
/// 
/// ```rust
/// use monkey_tail::prelude::*;
/// 
/// let sensors = SensorEnvironment::new();
/// let config = NoiseReductionConfig::default();
/// let extractor = TrailExtractor::new(config);
/// let trails = extractor.extract_trails(&sensors).await?;
/// ```
/// 
/// # Errors
/// 
/// Returns `SensorError` if sensor readings fail or `ExtractionError` if
/// pattern extraction cannot converge within the specified tolerance.
pub async fn extract_trails(
    &self,
    sensors: &SensorEnvironment,
) -> Result<Vec<ThermodynamicTrail>, ExtractionError> {
    // Implementation
}
```

## Testing

### Test Categories

1. **Unit Tests**: Test individual functions and modules
2. **Integration Tests**: Test module interactions
3. **Benchmark Tests**: Performance validation
4. **Example Tests**: Verify example code works

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_trail_extraction_convergence() {
        // Arrange
        let sensor_data = create_test_sensor_data();
        let config = NoiseReductionConfig::default();
        let extractor = TrailExtractor::new(config);
        
        // Act
        let trails = extractor.extract_trails(&sensor_data).await.unwrap();
        
        // Assert
        assert!(!trails.is_empty());
        assert!(trails.iter().all(|t| t.coherence_score() > 0.5));
    }
    
    #[test]
    fn test_pattern_persistence() {
        let pattern = create_test_pattern();
        let thresholds = vec![
            NoiseThreshold::new(1.0),
            NoiseThreshold::new(0.5), 
            NoiseThreshold::new(0.1)
        ];
        
        assert!(pattern.is_persistent(&thresholds));
        assert!(pattern.persistence_score() > 0.8);
    }
}
```

### Running Tests

```bash
# All tests
make test

# Specific crate
cargo test -p monkey-tail-core

# With coverage
make coverage

# Benchmarks
make bench
```

## Architecture Guidelines

### Module Organization

Each crate should have clear responsibilities:

- **Core**: Fundamental types and traits
- **Sensors**: Hardware and data source integration
- **Trail Extraction**: Algorithm implementations
- **Identity**: Ephemeral identity construction
- **CLI**: User interface and configuration
- **Bindings**: Language bindings

### Error Handling

Use `anyhow` for application errors and `thiserror` for library errors:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SensorError {
    #[error("Sensor read timeout after {duration:?}")]
    ReadTimeout { duration: Duration },
    
    #[error("Invalid sensor configuration: {message}")]
    InvalidConfig { message: String },
    
    #[error("Hardware error: {0}")]
    Hardware(#[from] HardwareError),
}
```

### Performance Considerations

- Use `rayon` for CPU-intensive parallel operations
- Implement `Send + Sync` for multi-threaded usage
- Consider memory allocation patterns in hot paths
- Profile with `cargo bench` before optimizing

## Sensor Integration

### Adding New Sensor Types

1. Implement the `SensorStream` trait:

```rust
pub struct NewSensor {
    config: NewSensorConfig,
    // sensor-specific fields
}

#[async_trait]
impl SensorStream for NewSensor {
    type Data = NewSensorData;
    type Error = NewSensorError;
    
    fn dimensionality(&self) -> usize {
        // Return data dimensionality
    }
    
    async fn read(&mut self) -> Result<Self::Data, Self::Error> {
        // Implement sensor reading
    }
    
    async fn read_window(&mut self, duration: Duration) -> Result<Vec<Self::Data>, Self::Error> {
        // Implement windowed reading
    }
    
    fn noise_characteristics(&self) -> NoiseCharacteristics {
        // Return noise profile
    }
}
```

2. Add configuration options
3. Write comprehensive tests
4. Update documentation and examples
5. Add feature flag if optional

### Sensor Guidelines

- Handle hardware failures gracefully
- Implement proper cleanup in `Drop`
- Respect privacy and security requirements
- Provide mock implementations for testing
- Document hardware requirements and limitations

## Performance Optimization

### Profiling

```bash
# CPU profiling
make profile

# Memory profiling  
make memcheck

# Custom profiling
cargo build --release
perf record --call-graph=dwarf target/release/monkey-tail
perf report
```

### Optimization Checklist

- [ ] Minimize allocations in hot paths
- [ ] Use appropriate data structures
- [ ] Leverage SIMD when possible
- [ ] Consider cache locality
- [ ] Profile before and after changes
- [ ] Benchmark performance-critical changes

## Documentation

### Types of Documentation

1. **API Documentation**: Rustdoc comments
2. **User Guide**: README and usage examples
3. **Developer Guide**: This document
4. **White Paper**: Theoretical foundation
5. **Tutorials**: Step-by-step guides

### Building Documentation

```bash
# Generate docs
make docs

# With all features
make docs-full

# Serve locally
cargo doc --open
```

## Release Process

### Version Management

We use semantic versioning (SemVer):
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes

### Release Checklist

- [ ] Update version numbers in Cargo.toml files
- [ ] Update CHANGELOG.md
- [ ] Run full test suite (`make test-full`)
- [ ] Run security audit (`make audit`)
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in git
- [ ] Publish to crates.io

## Getting Help

### Communication Channels

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Email: kundai.sachikonye@wzw.tum.de for direct contact

### Issue Reporting

When reporting bugs, include:

1. Rust version and platform
2. Monkey-Tail version
3. Minimal reproduction case
4. Expected vs actual behavior
5. Relevant log output

### Feature Requests

For new features, provide:

1. Use case description
2. Proposed API design
3. Implementation considerations
4. Alternative solutions considered

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Documentation credits

Thank you for contributing to Monkey-Tail!