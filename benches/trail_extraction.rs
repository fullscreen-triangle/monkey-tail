use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use monkey_tail::prelude::*;
use std::time::Duration;

fn bench_noise_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise_reduction");
    
    // Test different data sizes
    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("progressive_reduction", size),
            size,
            |b, &size| {
                // Generate synthetic sensor data
                let sensor_data = generate_synthetic_sensor_data(size);
                let config = NoiseReductionConfig::default();
                
                b.iter(|| {
                    let extractor = TrailExtractor::new(config.clone());
                    black_box(extractor.extract_patterns_sync(&sensor_data))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_pattern_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_persistence");
    
    for pattern_count in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("persistence_check", pattern_count),
            pattern_count,
            |b, &pattern_count| {
                let patterns = generate_test_patterns(pattern_count);
                let thresholds = generate_noise_thresholds(20);
                
                b.iter(|| {
                    for pattern in &patterns {
                        black_box(pattern.is_persistent(&thresholds));
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_multi_modal_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_modal");
    
    group.bench_function("visual_audio_gps", |b| {
        b.iter(|| {
            let visual_data = generate_visual_data(1000);
            let audio_data = generate_audio_data(1000);
            let gps_data = generate_gps_data(1000);
            
            let integrated = black_box(integrate_sensor_streams(
                &visual_data,
                &audio_data, 
                &gps_data
            ));
            
            integrated
        });
    });
    
    group.finish();
}

fn bench_identity_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("identity_construction");
    
    for trail_count in [5, 10, 25, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("ephemeral_identity", trail_count),
            trail_count,
            |b, &trail_count| {
                let trails = generate_thermodynamic_trails(trail_count);
                
                b.iter(|| {
                    black_box(EphemeralIdentity::from_trails(trails.clone()))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_temporal_decay(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_decay");
    
    group.bench_function("identity_evolution", |b| {
        let mut identity = create_test_identity();
        let time_steps = 100;
        
        b.iter(|| {
            for _ in 0..time_steps {
                black_box(identity.apply_temporal_decay(Duration::from_secs(1)));
            }
        });
    });
    
    group.finish();
}

// Helper functions for benchmark data generation
fn generate_synthetic_sensor_data(size: usize) -> Vec<SensorData> {
    (0..size)
        .map(|i| SensorData::new(
            format!("sensor_{}", i % 4),
            vec![
                (i as f64 * 0.1).sin() + rand::random::<f64>() * 0.1,
                (i as f64 * 0.2).cos() + rand::random::<f64>() * 0.1,
                rand::random::<f64>(),
            ]
        ))
        .collect()
}

fn generate_test_patterns(count: usize) -> Vec<ThermodynamicPattern> {
    (0..count)
        .map(|i| ThermodynamicPattern::new(
            format!("pattern_{}", i),
            0.5 + (i as f64) / (count as f64) * 0.4, // Varying signal strength
            generate_pattern_data(32)
        ))
        .collect()
}

fn generate_noise_thresholds(count: usize) -> Vec<NoiseThreshold> {
    (0..count)
        .map(|i| NoiseThreshold::new(1.0 - (i as f64) / (count as f64)))
        .collect()
}

fn generate_visual_data(size: usize) -> Vec<f64> {
    (0..size).map(|_| rand::random()).collect()
}

fn generate_audio_data(size: usize) -> Vec<f64> {
    (0..size).map(|_| rand::random()).collect()
}

fn generate_gps_data(size: usize) -> Vec<(f64, f64)> {
    (0..size).map(|_| (rand::random(), rand::random())).collect()
}

fn generate_thermodynamic_trails(count: usize) -> Vec<ThermodynamicTrail> {
    (0..count)
        .map(|i| ThermodynamicTrail::new(
            format!("trail_{}", i),
            generate_trail_patterns(10),
            0.8 - (i as f64) * 0.01 // Decreasing coherence
        ))
        .collect()
}

fn generate_pattern_data(size: usize) -> Vec<f64> {
    (0..size).map(|_| rand::random()).collect()
}

fn generate_trail_patterns(count: usize) -> Vec<ThermodynamicPattern> {
    (0..count)
        .map(|i| ThermodynamicPattern::new(
            format!("trail_pattern_{}", i),
            0.6 + rand::random::<f64>() * 0.3,
            generate_pattern_data(16)
        ))
        .collect()
}

fn integrate_sensor_streams(
    visual: &[f64],
    audio: &[f64],
    gps: &[(f64, f64)]
) -> Vec<f64> {
    // Simple integration for benchmarking
    let min_len = visual.len().min(audio.len()).min(gps.len());
    (0..min_len)
        .map(|i| visual[i] * 0.4 + audio[i] * 0.4 + gps[i].0 * 0.2)
        .collect()
}

fn create_test_identity() -> EphemeralIdentity {
    let trails = generate_thermodynamic_trails(10);
    EphemeralIdentity::from_trails(trails).unwrap()
}

criterion_group!(
    benches,
    bench_noise_reduction,
    bench_pattern_persistence,
    bench_multi_modal_integration,
    bench_identity_construction,
    bench_temporal_decay
);
criterion_main!(benches);