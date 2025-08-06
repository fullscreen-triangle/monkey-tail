//! # Monkey-Tail: Ephemeral Digital Identity Framework
//! 
//! A framework for ephemeral digital identity through multi-modal thermodynamic trail extraction.
//! 
//! ## Overview
//! 
//! Monkey-Tail treats digital interaction patterns as thermodynamic trails naturally emergent 
//! from user behavior, analogous to animal tracking in natural environments. The framework 
//! extracts meaningful patterns from high-dimensional sensor data through progressive noise 
//! reduction without requiring precise measurements or comprehensive metric collection.
//! 
//! ## Core Concepts
//! 
//! - **Thermodynamic Trails**: Natural patterns in digital interaction behavior
//! - **Progressive Noise Reduction**: Algorithm for extracting patterns from sensor noise
//! - **Ephemeral Identity**: Temporal identity construction that evolves with behavior
//! - **Multi-Modal Sensors**: Integration of visual, audio, location, and biological data
//! 
//! ## Example Usage
//! 
//! ```rust,no_run
//! use monkey_tail::prelude::*;
//! 
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create sensor environment
//!     let mut sensors = SensorEnvironment::new();
//!     sensors.add_visual_sensor()?;
//!     sensors.add_audio_sensor()?;
//!     sensors.add_gps_sensor()?;
//!     
//!     // Initialize trail extractor
//!     let extractor = TrailExtractor::new(
//!         NoiseReductionConfig::default()
//!     );
//!     
//!     // Extract thermodynamic trails
//!     let trails = extractor.extract_trails(&sensors).await?;
//!     
//!     // Construct ephemeral identity
//!     let identity = EphemeralIdentity::from_trails(trails)?;
//!     
//!     println!("Identity coherence: {:.3}", identity.coherence_score());
//!     
//!     Ok(())
//! }
//! ```

pub use monkey_tail_core as core;
pub use monkey_tail_sensors as sensors;
pub use monkey_tail_trail_extraction as trail_extraction;
pub use monkey_tail_identity as identity;

/// Convenient re-exports for common usage
pub mod prelude {
    pub use crate::core::{
        SensorStream, ThermodynamicTrail, NoiseThreshold, 
        PatternPersistence, SensorEnvironment
    };
    pub use crate::sensors::{
        VisualSensor, AudioSensor, GpsSensor, BiologicalSensor,
        SensorData, SensorError
    };
    pub use crate::trail_extraction::{
        TrailExtractor, NoiseReductionConfig, ProgressiveNoiseReduction,
        PatternExtractor, ThermodynamicPattern
    };
    pub use crate::identity::{
        EphemeralIdentity, IdentityBuilder, TemporalDecay,
        IdentityCoherence, PrivacyPreservation
    };
    pub use anyhow::{Result, Error};
}