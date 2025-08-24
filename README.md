<h1 align="center">Monkey Tail</h1>
<p align="center"><em>If having a tail was compulsory, we would settle on the one that belongs to a monkey</em></p>

<p align="center">
  <img src="assets/images/balint-miko-YGz4PDiv36E-unsplash.jpg" alt="Monkey Tail" width="500"/>
</p>

A framework for ephemeral digital identity through multi-modal thermodynamic trail extraction.

## Overview

Monkey-Tail treats digital interaction patterns as thermodynamic trails naturally emergent from user behavior, analogous to animal tracking in natural environments. The framework extracts meaningful patterns from high-dimensional sensor data through progressive noise reduction without requiring precise measurements or comprehensive metric collection.

## Core Concepts

- **Thermodynamic Trails**: Natural patterns in digital interaction behavior that emerge from noise
- **Progressive Noise Reduction**: Algorithm for extracting persistent patterns by gradually reducing noise thresholds
- **Ephemeral Identity**: Temporal identity construction that evolves with behavior and naturally decays
- **Multi-Modal Sensors**: Integration of visual, audio, location, biological, and interaction data streams

## Architecture

The framework is organized into several modular crates:

- **`monkey-tail-core`**: Core types, traits, and fundamental building blocks
- **`monkey-tail-sensors`**: Multi-modal sensor integration (visual, audio, GPS, biological)
- **`monkey-tail-trail-extraction`**: Progressive noise reduction and pattern extraction algorithms
- **`monkey-tail-identity`**: Ephemeral identity construction and temporal decay management
- **`monkey-tail-cli`**: Command-line interface for the framework
- **`monkey-tail-bindings`**: Python and WebAssembly bindings

## Quick Start

### Installation

```bash
git clone https://github.com/fullscreen-triangle/monkey-tail.git
cd monkey-tail
cargo build --release
```

### Basic Usage

```rust
use monkey_tail::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create sensor environment
    let mut sensors = SensorEnvironment::new();
    sensors.add_visual_sensor()?;
    sensors.add_audio_sensor()?;
    sensors.add_gps_sensor()?;

    // Initialize trail extractor with progressive noise reduction
    let extractor = TrailExtractor::new(
        NoiseReductionConfig::default()
    );

    // Extract thermodynamic trails from sensor noise
    let trails = extractor.extract_trails(&sensors).await?;

    // Construct ephemeral identity from extracted trails
    let identity = EphemeralIdentity::from_trails(trails)?;

    println!("Identity coherence: {:.3}", identity.coherence_score());
    println!("Active patterns: {}", identity.pattern_count());

    Ok(())
}
```

### Command Line Interface

```bash
# Start real-time trail extraction
cargo run -- extract --sensors visual,audio,gps --output trails.json

# Analyze existing sensor data
cargo run -- analyze --input sensor_data.bin --config analysis.toml

# Generate identity from trails
cargo run -- identity --trails trails.json --decay-rate 0.1
```

## Configuration

Create a `config.toml` file:

```toml
[sensors]
sampling_rate = 10.0  # Hz
buffer_size = 1000
parallel_reading = true

[noise_reduction]
max_threshold = 1.0
min_threshold = 0.1
reduction_step = 0.05
convergence_tolerance = 0.001

[identity]
decay_rate = 0.1
temporal_window = 3600  # seconds
coherence_threshold = 0.7

[visual]
enable = true
confidence_threshold = 0.8

[audio]
enable = true
sample_rate = 44100
chunk_size = 1024

[gps]
enable = true
accuracy_threshold = 10.0  # meters
```

## Features

### Multi-Modal Sensor Support

- **Visual Processing**: Eye tracking, image preference, visual attention patterns
- **Audio Analysis**: Music preference, rhythm detection, ambient sound tolerance
- **Geolocation**: Movement patterns, location preferences, trajectory analysis
- **Biological Data**: Genomic variants, metabolomic profiles, circadian rhythms
- **Interaction Patterns**: Keystroke dynamics, mouse movements, navigation behavior

### Advanced Algorithms

- **Progressive Noise Reduction**: Extract meaningful patterns from high-dimensional sensor noise
- **Pattern Persistence Detection**: Identify patterns that remain stable across noise thresholds
- **Temporal Identity Evolution**: Natural identity adaptation with configurable decay
- **Privacy-Preserving Extraction**: Noise-based processing without explicit data collection

### Performance Optimizations

- **Parallel Processing**: Multi-threaded sensor reading and pattern extraction
- **Memory Efficiency**: Streaming processing for large sensor data volumes
- **Hardware Acceleration**: Optional GPU processing for intensive computations
- **Adaptive Quality**: Dynamic adjustment based on available computational resources

## Mathematical Foundation

The framework implements the theoretical model described in our [white paper](docs/monkey-tail.tex):

```
T_u(E, θ) = {p ∈ P : SNR(p, E) > θ}
```

Where:

- `T_u` is the thermodynamic trail for user `u`
- `E` is the sensor environment
- `θ` is the noise threshold
- `P` is the pattern space
- `SNR(p, E)` is the signal-to-noise ratio of pattern `p`

## Development

### Building

```bash
# Build all crates
cargo build

# Build with all sensor features
cargo build --features all-sensors

# Release build with optimizations
cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p monkey-tail-core

# Run benchmarks
cargo bench
```

### Linting

```bash
# Format code
cargo fmt

# Run clippy
cargo clippy -- -D warnings

# Run with all features
cargo clippy --all-features -- -D warnings
```

## Integration

### Python Bindings

```python
import monkey_tail

# Initialize sensor environment
sensors = monkey_tail.SensorEnvironment()
sensors.add_visual_sensor()
sensors.add_audio_sensor()

# Extract trails
extractor = monkey_tail.TrailExtractor()
trails = extractor.extract_trails(sensors)

# Build identity
identity = monkey_tail.EphemeralIdentity.from_trails(trails)
print(f"Coherence: {identity.coherence_score()}")
```

### WebAssembly

```javascript
import init, { SensorEnvironment, TrailExtractor } from "./pkg/monkey_tail.js";

async function run() {
  await init();

  const sensors = new SensorEnvironment();
  const extractor = new TrailExtractor();

  const trails = await extractor.extract_trails(sensors);
  console.log(`Extracted ${trails.length} trails`);
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{monkey_tail_2024,
  title = {Monkey-Tail: Ephemeral Digital Identity Through Thermodynamic Trail Extraction},
  author = {Sachikonye, Kundai Farai},
  year = {2024},
  url = {https://github.com/fullscreen-triangle/monkey-tail}
}
```

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) and submit pull requests to the main branch.

## Contact

For questions and support, please open an issue on GitHub or contact:

- Kundai Farai Sachikonye (kundai.sachikonye@wzw.tum.de)
