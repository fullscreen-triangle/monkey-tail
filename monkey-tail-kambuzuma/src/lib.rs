use monkey_tail_core::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use std::sync::Arc;
use anyhow::Result;
use tracing::{info, debug, warn};

pub mod processor;
pub mod bmd;
pub mod adaptation;

pub use processor::*;
pub use bmd::*;
pub use adaptation::*;

/// Personalized input for Kambuzuma processing with semantic identity context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedStageInput {
    pub query: String,
    pub context: Option<String>,
    pub user_semantic_identity: SemanticIdentity,
    pub interaction_data: InteractionData,
    pub environmental_context: EnvironmentalContext,
    pub timestamp: DateTime<Utc>,
}

/// Interaction data for personalized processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionData {
    pub query_complexity: f64,
    pub domain_context: String,
    pub user_expertise_level: f64,
    pub communication_preferences: CommunicationPreferences,
    pub urgency_level: UrgencyLevel,
    pub expected_response_type: ResponseType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPreferences {
    pub detail_level: DetailLevel,
    pub style: CommunicationStyle,
    pub include_examples: bool,
    pub include_explanations: bool,
    pub technical_depth: TechnicalDepth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseType {
    DirectAnswer,
    Explanation,
    Tutorial,
    Analysis,
    Creative,
    Troubleshooting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TechnicalDepth {
    Novice,      // Simple, non-technical language
    Intermediate, // Some technical terms with explanations
    Advanced,    // Technical language assumed
    Expert,      // Full technical precision
}

/// Personalized processing result with BMD effectiveness and adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedProcessingResult {
    pub content: String,
    pub bmd_effectiveness: f64,
    pub response_adaptation: ResponseAdaptation,
    pub learning_insights: Vec<LearningInsight>,
    pub confidence_score: f64,
    pub processing_time_ms: u64,
    pub metadata: ProcessingMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseAdaptation {
    pub technical_depth: TechnicalDepth,
    pub adapted_content: String,
    pub interaction_suggestions: Vec<String>,
    pub follow_up_questions: Vec<String>,
    pub estimated_comprehension: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningInsight {
    pub domain: String,
    pub insight_type: InsightType,
    pub confidence: f64,
    pub description: String,
    pub suggested_next_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    KnowledgeGap,
    Misconception,
    StrengthArea,
    LearningOpportunity,
    ConceptualConnection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub bmd_frames_considered: u32,
    pub s_entropy_navigation_steps: u32,
    pub adaptation_iterations: u32,
    pub domain_expertise_detected: HashMap<String, f64>,
}

/// Main personalized Kambuzuma processor
pub struct PersonalizedKambuzumaProcessor {
    // Mock Kambuzuma processor - in real implementation this would interface with actual Kambuzuma
    bmd_orchestrator: Arc<RwLock<BmdOrchestrator>>,
    response_adapter: Arc<RwLock<ResponseAdapter>>,
    learning_analyzer: Arc<RwLock<LearningAnalyzer>>,
    processing_stats: Arc<RwLock<ProcessingStats>>,
}

#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub total_queries: u64,
    pub average_bmd_effectiveness: f64,
    pub average_processing_time_ms: u64,
    pub domain_query_counts: HashMap<String, u64>,
}

impl PersonalizedKambuzumaProcessor {
    pub async fn new() -> Result<Self> {
        info!("Initializing PersonalizedKambuzumaProcessor");
        
        Ok(Self {
            bmd_orchestrator: Arc::new(RwLock::new(BmdOrchestrator::new())),
            response_adapter: Arc::new(RwLock::new(ResponseAdapter::new())),
            learning_analyzer: Arc::new(RwLock::new(LearningAnalyzer::new())),
            processing_stats: Arc::new(RwLock::new(ProcessingStats::default())),
        })
    }

    /// Process query with semantic identity enhancement
    pub async fn process_query_with_semantic_identity(
        &mut self,
        user_id: Uuid,
        input: PersonalizedStageInput,
    ) -> Result<PersonalizedProcessingResult> {
        let start_time = std::time::Instant::now();
        
        debug!("Processing query for user {}: {}", user_id, input.query);

        // Calculate BMD effectiveness for this domain
        let bmd_effectiveness = input.user_semantic_identity
            .calculate_bmd_effectiveness(&input.interaction_data.domain_context);

        debug!("BMD effectiveness for domain '{}': {:.2}%", 
               input.interaction_data.domain_context, bmd_effectiveness * 100.0);

        // Process through BMD orchestrator with enhanced effectiveness
        let bmd_result = {
            let mut orchestrator = self.bmd_orchestrator.write().await;
            orchestrator.process_with_bmd_effectiveness(&input, bmd_effectiveness).await?
        };

        // Adapt response based on user communication patterns and expertise
        let response_adaptation = {
            let mut adapter = self.response_adapter.write().await;
            adapter.adapt_response_for_user(
                &bmd_result,
                &input.user_semantic_identity,
                &input.interaction_data,
            ).await?
        };

        // Extract learning insights for identity evolution
        let learning_insights = {
            let mut analyzer = self.learning_analyzer.write().await;
            analyzer.extract_learning_insights(
                &input.query,
                &bmd_result,
                &input.user_semantic_identity,
            ).await?
        };

        let processing_time = start_time.elapsed();
        
        // Update processing statistics
        {
            let mut stats = self.processing_stats.write().await;
            stats.total_queries += 1;
            stats.average_bmd_effectiveness = 
                (stats.average_bmd_effectiveness * (stats.total_queries - 1) as f64 + bmd_effectiveness) 
                / stats.total_queries as f64;
            stats.average_processing_time_ms = 
                (stats.average_processing_time_ms * (stats.total_queries - 1) + processing_time.as_millis() as u64) 
                / stats.total_queries;
            
            *stats.domain_query_counts.entry(input.interaction_data.domain_context.clone()).or_insert(0) += 1;
        }

        let result = PersonalizedProcessingResult {
            content: response_adaptation.adapted_content.clone(),
            bmd_effectiveness,
            response_adaptation,
            learning_insights,
            confidence_score: bmd_result.confidence,
            processing_time_ms: processing_time.as_millis() as u64,
            metadata: ProcessingMetadata {
                bmd_frames_considered: bmd_result.frames_considered,
                s_entropy_navigation_steps: bmd_result.navigation_steps,
                adaptation_iterations: 1, // Simplified for now
                domain_expertise_detected: input.user_semantic_identity.understanding_vector.domains.clone(),
            },
        };

        info!("Query processed successfully in {}ms with {:.1}% BMD effectiveness", 
              processing_time.as_millis(), bmd_effectiveness * 100.0);

        Ok(result)
    }

    /// Get current processing statistics
    pub async fn get_processing_stats(&self) -> ProcessingStats {
        self.processing_stats.read().await.clone()
    }

    /// Get ecosystem signature for security
    pub fn get_ecosystem_signature(&self) -> EcosystemSignature {
        EcosystemSignature::generate_unique()
    }

    /// Get current person signature for the user
    pub async fn get_current_person_signature(&self, _user_id: Uuid) -> Result<PersonSignature> {
        // In a real implementation, this would extract the person signature
        // from current behavioral patterns and interaction history
        Ok(PersonSignature::default())
    }
}

impl Default for CommunicationPreferences {
    fn default() -> Self {
        Self {
            detail_level: DetailLevel::Moderate,
            style: CommunicationStyle::Direct,
            include_examples: true,
            include_explanations: true,
            technical_depth: TechnicalDepth::Intermediate,
        }
    }
}

impl From<CommunicationPreferences> for CommunicationPatterns {
    fn from(prefs: CommunicationPreferences) -> Self {
        CommunicationPatterns {
            preferred_detail_level: prefs.detail_level,
            communication_style: prefs.style,
            learning_style: LearningStyle::default(),
            interaction_preferences: InteractionPreferences::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_personalized_processor_creation() {
        let processor = PersonalizedKambuzumaProcessor::new().await;
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_bmd_effectiveness_scaling() {
        let mut processor = PersonalizedKambuzumaProcessor::new().await.unwrap();
        
        // Create test input with novice user
        let mut identity = SemanticIdentity::new();
        let input = PersonalizedStageInput {
            query: "What is quantum mechanics?".to_string(),
            context: None,
            user_semantic_identity: identity.clone(),
            interaction_data: InteractionData {
                query_complexity: 0.7,
                domain_context: "physics".to_string(),
                user_expertise_level: 0.1, // Novice
                communication_preferences: CommunicationPreferences::default(),
                urgency_level: UrgencyLevel::Medium,
                expected_response_type: ResponseType::Explanation,
            },
            environmental_context: EnvironmentalContext::default(),
            timestamp: Utc::now(),
        };

        let result = processor.process_query_with_semantic_identity(
            Uuid::new_v4(), 
            input
        ).await.unwrap();

        // Should have base BMD effectiveness for novice
        assert!(result.bmd_effectiveness >= 0.6);
        assert!(result.bmd_effectiveness <= 0.75);
        
        // Now test with expert user
        identity.understanding_vector.domains.insert("physics".to_string(), 0.9);
        let expert_input = PersonalizedStageInput {
            query: "Derive the Schrödinger equation".to_string(),
            context: None,
            user_semantic_identity: identity,
            interaction_data: InteractionData {
                query_complexity: 0.9,
                domain_context: "physics".to_string(),
                user_expertise_level: 0.9, // Expert
                communication_preferences: CommunicationPreferences {
                    technical_depth: TechnicalDepth::Expert,
                    ..Default::default()
                },
                urgency_level: UrgencyLevel::Medium,
                expected_response_type: ResponseType::Analysis,
            },
            environmental_context: EnvironmentalContext::default(),
            timestamp: Utc::now(),
        };

        let expert_result = processor.process_query_with_semantic_identity(
            Uuid::new_v4(), 
            expert_input
        ).await.unwrap();

        // Should have much higher BMD effectiveness for expert
        assert!(expert_result.bmd_effectiveness >= 0.85);
        assert!(expert_result.bmd_effectiveness <= 0.95);
        assert!(expert_result.bmd_effectiveness > result.bmd_effectiveness);
    }

    #[tokio::test]
    async fn test_processing_stats_tracking() {
        let mut processor = PersonalizedKambuzumaProcessor::new().await.unwrap();
        
        let input = PersonalizedStageInput {
            query: "Test query".to_string(),
            context: None,
            user_semantic_identity: SemanticIdentity::new(),
            interaction_data: InteractionData {
                query_complexity: 0.5,
                domain_context: "test".to_string(),
                user_expertise_level: 0.5,
                communication_preferences: CommunicationPreferences::default(),
                urgency_level: UrgencyLevel::Medium,
                expected_response_type: ResponseType::DirectAnswer,
            },
            environmental_context: EnvironmentalContext::default(),
            timestamp: Utc::now(),
        };

        // Process a few queries
        for _ in 0..3 {
            processor.process_query_with_semantic_identity(
                Uuid::new_v4(), 
                input.clone()
            ).await.unwrap();
        }

        let stats = processor.get_processing_stats().await;
        assert_eq!(stats.total_queries, 3);
        assert!(stats.average_bmd_effectiveness > 0.0);
        assert!(stats.domain_query_counts.contains_key("test"));
        assert_eq!(stats.domain_query_counts["test"], 3);
    }
}
