use thiserror::Error;

#[derive(Error, Debug)]
pub enum MonkeyTailError {
    #[error("Semantic identity error: {0}")]
    SemanticIdentity(String),
    
    #[error("BMD processing error: {0}")]
    BmdProcessing(String),
    
    #[error("Competency assessment error: {0}")]
    CompetencyAssessment(String),
    
    #[error("Ecosystem security violation: {0}")]
    EcosystemSecurity(String),
    
    #[error("Temporal context error: {0}")]
    TemporalContext(String),
    
    #[error("Communication pattern error: {0}")]
    CommunicationPattern(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("UUID error: {0}")]
    Uuid(#[from] uuid::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, MonkeyTailError>;
