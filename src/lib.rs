//! # Monkey-Tail: Ephemeral Semantic Digital Identity System
//!
//! Monkey-Tail is a revolutionary framework for creating consciousness-aware AI systems
//! through ephemeral semantic identity and noise-to-meaning extraction.
//!
//! ## Core Principles
//!
//! - **One Machine, One User, One Application**: Dedicated personal AI systems
//! - **Ephemeral Identity**: No stored personal data, identity exists only as current AI understanding
//! - **Ecosystem Security**: Security through uniqueness, not computational complexity
//! - **Semantic Competency**: Real-time assessment of user expertise across domains
//! - **BMD Effectiveness**: Biological Maxwell Demon processing scales with user understanding

pub use monkey_tail_core::*;
pub use monkey_tail_sensors::*;
pub use monkey_tail_identity::*;
pub use monkey_tail_kambuzuma::*;
pub use monkey_tail_competency::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        SemanticIdentity,
        PersonalizedKambuzumaProcessor,
        EphemeralIdentityProcessor,
        FourSidedTriangleAssessor,
    };
    
    pub use uuid::Uuid;
    pub use chrono::{DateTime, Utc};
    pub use anyhow::{Result, Error};
}