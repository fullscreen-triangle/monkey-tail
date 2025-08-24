use anyhow::Result;
use tracing::{info, warn};
use tracing_subscriber;

use monkey_tail_core::SemanticIdentity;
use monkey_tail_kambuzuma::PersonalizedKambuzumaProcessor;
use monkey_tail_identity::EphemeralIdentityProcessor;
use monkey_tail_competency::FourSidedTriangleAssessor;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🐒 Monkey-Tail: Ephemeral Semantic Digital Identity System");
    info!("🧠 Integrating with Kambuzuma Neural Stack");
    info!("🔒 One Machine, One User, One Application");
    
    // Initialize the integrated system
    let mut processor = PersonalizedKambuzumaProcessor::new().await?;
    
    info!("✅ Monkey-Tail + Kambuzuma integration initialized successfully");
    info!("🚀 Ready for consciousness-aware AI processing");
    
    // Keep the system running
    tokio::signal::ctrl_c().await?;
    info!("👋 Shutting down Monkey-Tail system");
    
    Ok(())
}