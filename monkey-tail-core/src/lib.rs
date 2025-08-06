//! Core types and traits for the Monkey-Tail ephemeral identity framework.
//! 
//! This crate provides the fundamental building blocks for thermodynamic trail extraction
//! and ephemeral identity construction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use nalgebra::{DVector, DMatrix};
use anyhow::Result;

pub mod sensor;
pub mod trail;
pub mod pattern;
pub mod environment;
pub mod errors;

pub use sensor::*;
pub use trail::*;
pub use pattern::*;
pub use environment::*;
pub use errors::*;

/// Core trait for all sensor streams in the Monkey-Tail framework
pub trait SensorStream: Send + Sync {
    type Data: Clone + Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Get the dimensionality of this sensor stream
    fn dimensionality(&self) -> usize;
    
    /// Get the current sensor reading
    async fn read(&mut self) -> Result<Self::Data, Self::Error>;
    
    /// Get multiple readings over a time window
    async fn read_window(&mut self, duration: std::time::Duration) -> Result<Vec<Self::Data>, Self::Error>;
    
    /// Get the noise characteristics for this sensor
    fn noise_characteristics(&self) -> NoiseCharacteristics;
}

/// Noise characteristics for a sensor stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseCharacteristics {
    /// Standard deviation of sensor noise
    pub std_dev: f64,
    /// Frequency characteristics of noise
    pub frequency_profile: Vec<f64>,
    /// Signal-to-noise ratio
    pub snr: f64,
}

/// Represents a noise threshold for pattern extraction
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NoiseThreshold(pub f64);

impl NoiseThreshold {
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }
    
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Trait for measuring pattern persistence across noise thresholds
pub trait PatternPersistence {
    /// Check if a pattern persists across multiple noise thresholds
    fn is_persistent(&self, thresholds: &[NoiseThreshold]) -> bool;
    
    /// Get the persistence score (0.0 to 1.0)
    fn persistence_score(&self) -> f64;
    
    /// Get the minimum threshold at which this pattern appears
    fn minimum_threshold(&self) -> NoiseThreshold;
}

/// Configuration for the sensor environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorEnvironmentConfig {
    /// Sampling rate for continuous sensors (Hz)
    pub sampling_rate: f64,
    /// Buffer size for sensor data
    pub buffer_size: usize,
    /// Timeout for sensor readings
    pub read_timeout: std::time::Duration,
    /// Enable parallel sensor reading
    pub parallel_reading: bool,
}

impl Default for SensorEnvironmentConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 10.0, // 10 Hz default
            buffer_size: 1000,
            read_timeout: std::time::Duration::from_secs(1),
            parallel_reading: true,
        }
    }
}