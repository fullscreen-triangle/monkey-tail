use monkey_tail_core::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};
use tokio::sync::RwLock;
use std::sync::Arc;
use anyhow::Result;
use tracing::{debug, info, warn};

pub mod processor;
pub mod security;
pub mod patterns;

pub use processor::*;
pub use security::*;
pub use patterns::*;

/// Ephemeral identity processor that maintains no persistent storage
/// Identity exists only as current AI understanding
pub struct EphemeralIdentityProcessor {
    // Current session observations (ephemeral, in-memory only)
    current_observations: Arc<RwLock<HashMap<Uuid, CurrentObservations>>>,
    
    // Personality models (what AI currently understands about users)
    personality_models: Arc<RwLock<HashMap<Uuid, PersonalityModel>>>,
    
    // Environmental context tracking
    environmental_context: Arc<RwLock<HashMap<Uuid, EnvironmentalContext>>>,
    
    // Machine ecosystem signature for security
    machine_ecosystem_signature: EcosystemSignature,
    
    // Security validator
    security_validator: SecurityValidator,
    
    // Pattern analyzer for behavioral understanding
    pattern_analyzer: PatternAnalyzer,
}

/// Current observations about a user (ephemeral, session-only)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentObservations {
    pub session_start: DateTime<Utc>,
    pub interaction_patterns: InteractionPatterns,
    pub query_complexity_history: Vec<f64>,
    pub domain_engagement: HashMap<String, f64>,
    pub communication_preferences: CommunicationPreferences,
    pub learning_progression: LearningProgression,
    pub temporal_patterns: TemporalPatterns,
    pub last_update: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPatterns {
    pub response_times: Vec<u64>, // milliseconds
    pub query_lengths: Vec<usize>,
    pub interaction_frequency: f64, // interactions per hour
    pub session_durations: Vec<u64>, // minutes
    pub preferred_interaction_types: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProgression {
    pub domains_explored: Vec<String>,
    pub complexity_progression: HashMap<String, Vec<f64>>,
    pub understanding_improvements: HashMap<String, f64>,
    pub learning_velocity: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatterns {
    pub active_hours: Vec<u8>, // Hours of day when active
    pub session_patterns: Vec<SessionPattern>,
    pub attention_span_estimates: Vec<f64>, // minutes
    pub break_patterns: Vec<u64>, // minutes between sessions
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionPattern {
    pub start_time: DateTime<Utc>,
    pub duration_minutes: u64,
    pub interaction_count: u32,
    pub domains_covered: Vec<String>,
    pub engagement_level: f64,
}

/// Personality model representing what AI currently understands about user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityModel {
    pub what_ai_understands: SemanticIdentity,
    pub confidence_levels: HashMap<String, f64>, // Domain -> confidence in understanding
    pub interaction_history_summary: InteractionSummary,
    pub learning_style_assessment: LearningStyleAssessment,
    pub motivation_indicators: MotivationIndicators,
    pub behavioral_consistency: f64, // How consistent user behavior is
    pub model_last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStyleAssessment {
    pub prefers_examples: f64,
    pub prefers_theory_first: f64,
    pub prefers_hands_on: f64,
    pub prefers_visual_aids: f64,
    pub prefers_step_by_step: f64,
    pub learns_from_mistakes: f64,
    pub assessment_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationIndicators {
    pub curiosity_driven: f64,
    pub goal_oriented: f64,
    pub problem_solving_focused: f64,
    pub knowledge_seeking: f64,
    pub practical_application_focused: f64,
    pub social_learning_preference: f64,
}

impl EphemeralIdentityProcessor {
    pub fn new() -> Self {
        info!("Initializing EphemeralIdentityProcessor with zero persistent storage");
        
        Self {
            current_observations: Arc::new(RwLock::new(HashMap::new())),
            personality_models: Arc::new(RwLock::new(HashMap::new())),
            environmental_context: Arc::new(RwLock::new(HashMap::new())),
            machine_ecosystem_signature: EcosystemSignature::generate_unique(),
            security_validator: SecurityValidator::new(),
            pattern_analyzer: PatternAnalyzer::new(),
        }
    }

    /// Extract current semantic identity (ephemeral, no storage)
    /// This is the core of the "what AI understands about user" approach
    pub async fn extract_current_identity(
        &mut self,
        user_id: Uuid,
        interaction_data: &InteractionData,
    ) -> Result<SemanticIdentity> {
        debug!("Extracting ephemeral identity for user {}", user_id);

        // Update current observations from this interaction
        self.update_current_observations(user_id, interaction_data).await?;

        // Get or create personality model (what AI currently understands)
        let personality_model = self.get_or_create_personality_model(user_id).await?;

        // Extract semantic identity from current understanding
        let mut semantic_identity = personality_model.what_ai_understands.clone();

        // Update based on current interaction patterns
        self.update_semantic_identity_from_interaction(
            &mut semantic_identity,
            user_id,
            interaction_data,
        ).await?;

        // Validate ecosystem security through uniqueness
        self.validate_ecosystem_security(user_id, &semantic_identity).await?;

        debug!("Ephemeral identity extracted with BMD effectiveness: {:.2}% for domain '{}'", 
               semantic_identity.calculate_bmd_effectiveness(&interaction_data.domain_context) * 100.0,
               interaction_data.domain_context);

        Ok(semantic_identity)
    }

    /// Update identity based on interaction (ephemeral learning)
    pub async fn update_identity_from_interaction(
        &mut self,
        user_id: Uuid,
        query: &str,
        result: &PersonalizedProcessingResult,
        learning_insights: &[LearningInsight],
    ) -> Result<()> {
        debug!("Updating ephemeral identity from interaction feedback");

        // Update current observations
        {
            let mut observations = self.current_observations.write().await;
            if let Some(obs) = observations.get_mut(&user_id) {
                obs.update_from_interaction_result(query, result, learning_insights)?;
            }
        }

        // Update personality model based on learning insights
        {
            let mut models = self.personality_models.write().await;
            if let Some(model) = models.get_mut(&user_id) {
                model.update_from_learning_insights(learning_insights)?;
                model.update_confidence_from_result(result)?;
                model.model_last_updated = Utc::now();
            }
        }

        // Analyze patterns for behavioral understanding
        self.pattern_analyzer.analyze_interaction_patterns(user_id, query, result).await?;

        // No persistent storage - everything remains ephemeral
        Ok(())
    }

    /// Validate ecosystem security through uniqueness (zero computational overhead)
    async fn validate_ecosystem_security(
        &self,
        user_id: Uuid,
        semantic_identity: &SemanticIdentity,
    ) -> Result<()> {
        // Calculate person signature from current behavioral patterns
        let person_signature = self.calculate_person_signature(user_id, semantic_identity).await?;
        
        // Get machine signature
        let machine_signature = &self.machine_ecosystem_signature;

        // Security through ecosystem uniqueness (not cryptography)
        let ecosystem_uniqueness = self.security_validator.calculate_ecosystem_uniqueness(
            &person_signature,
            machine_signature,
        )?;

        if ecosystem_uniqueness < 0.95 {
            warn!("Ecosystem security threshold not met: {:.2}%", ecosystem_uniqueness * 100.0);
            return Err(MonkeyTailError::EcosystemSecurity(
                format!("Ecosystem uniqueness below threshold: {:.2}%", ecosystem_uniqueness * 100.0)
            ).into());
        }

        debug!("Ecosystem security validated: {:.2}% uniqueness", ecosystem_uniqueness * 100.0);
        Ok(())
    }

    async fn update_current_observations(
        &mut self,
        user_id: Uuid,
        interaction_data: &InteractionData,
    ) -> Result<()> {
        let mut observations = self.current_observations.write().await;
        
        let obs = observations.entry(user_id).or_insert_with(|| CurrentObservations {
            session_start: Utc::now(),
            interaction_patterns: InteractionPatterns::default(),
            query_complexity_history: Vec::new(),
            domain_engagement: HashMap::new(),
            communication_preferences: interaction_data.communication_preferences.clone(),
            learning_progression: LearningProgression::default(),
            temporal_patterns: TemporalPatterns::default(),
            last_update: Utc::now(),
        });

        // Update interaction patterns
        obs.query_complexity_history.push(interaction_data.query_complexity);
        
        // Update domain engagement
        let current_engagement = obs.domain_engagement
            .get(&interaction_data.domain_context)
            .unwrap_or(&0.0);
        obs.domain_engagement.insert(
            interaction_data.domain_context.clone(),
            (current_engagement + 1.0).min(10.0), // Cap at 10 for normalization
        );

        // Update temporal patterns
        let now = Utc::now();
        obs.temporal_patterns.active_hours.push(now.hour() as u8);
        obs.last_update = now;

        Ok(())
    }

    async fn get_or_create_personality_model(&mut self, user_id: Uuid) -> Result<PersonalityModel> {
        let mut models = self.personality_models.write().await;
        
        if let Some(model) = models.get(&user_id) {
            Ok(model.clone())
        } else {
            let new_model = PersonalityModel {
                what_ai_understands: SemanticIdentity::new(),
                confidence_levels: HashMap::new(),
                interaction_history_summary: InteractionSummary::default(),
                learning_style_assessment: LearningStyleAssessment::default(),
                motivation_indicators: MotivationIndicators::default(),
                behavioral_consistency: 0.5, // Start neutral
                model_last_updated: Utc::now(),
            };
            
            models.insert(user_id, new_model.clone());
            Ok(new_model)
        }
    }

    async fn update_semantic_identity_from_interaction(
        &self,
        semantic_identity: &mut SemanticIdentity,
        user_id: Uuid,
        interaction_data: &InteractionData,
    ) -> Result<()> {
        // Update understanding based on query complexity and domain
        let domain = &interaction_data.domain_context;
        let complexity = interaction_data.query_complexity;
        let expertise = interaction_data.user_expertise_level;

        // Estimate interaction success based on complexity vs expertise match
        let interaction_success = 1.0 - (complexity - expertise).abs();
        
        // Update domain understanding
        semantic_identity.update_domain_understanding(domain, interaction_success, complexity);

        // Update communication patterns
        semantic_identity.communication_patterns = interaction_data.communication_preferences.clone().into();

        // Update temporal context
        semantic_identity.temporal_context.current_session_start = Some(Utc::now());
        
        // Update emotional state based on interaction
        self.update_emotional_state(semantic_identity, interaction_data).await?;

        Ok(())
    }

    async fn update_emotional_state(
        &self,
        semantic_identity: &mut SemanticIdentity,
        interaction_data: &InteractionData,
    ) -> Result<()> {
        let emotional_state = &mut semantic_identity.emotional_state;
        
        // Update engagement based on query complexity and domain familiarity
        let domain_familiarity = semantic_identity.understanding_vector.domains
            .get(&interaction_data.domain_context)
            .unwrap_or(&0.0);
        
        let engagement = if interaction_data.query_complexity > *domain_familiarity + 0.2 {
            0.7 // Challenging but not overwhelming
        } else if interaction_data.query_complexity < *domain_familiarity - 0.2 {
            0.4 // Too easy, lower engagement
        } else {
            0.9 // Good match, high engagement
        };
        
        emotional_state.engagement_level = engagement;
        
        // Update confidence based on domain understanding
        emotional_state.confidence_level = *domain_familiarity;
        
        // Update curiosity based on learning opportunities
        emotional_state.curiosity_level = (1.0 - domain_familiarity).max(0.3);

        Ok(())
    }

    async fn calculate_person_signature(
        &self,
        user_id: Uuid,
        semantic_identity: &SemanticIdentity,
    ) -> Result<PersonSignature> {
        let observations = self.current_observations.read().await;
        
        let behavioral_patterns = if let Some(obs) = observations.get(&user_id) {
            let mut patterns = HashMap::new();
            
            // Extract behavioral patterns from observations
            patterns.insert("avg_query_complexity".to_string(), 
                           obs.query_complexity_history.iter().sum::<f64>() / obs.query_complexity_history.len().max(1) as f64);
            
            patterns.insert("interaction_frequency".to_string(), obs.interaction_patterns.interaction_frequency);
            
            patterns.insert("domain_diversity".to_string(), obs.domain_engagement.len() as f64 / 10.0);
            
            patterns.insert("session_consistency".to_string(), 
                           obs.temporal_patterns.session_patterns.len() as f64 / 24.0);
            
            patterns
        } else {
            HashMap::new()
        };

        let interaction_style = format!("{:?}", semantic_identity.communication_patterns.communication_style);
        
        let preference_consistency = semantic_identity.understanding_vector.domains.len() as f64 / 20.0;
        
        let temporal_patterns = semantic_identity.temporal_context.time_of_day_patterns
            .values()
            .cloned()
            .collect();

        Ok(PersonSignature {
            behavioral_patterns,
            interaction_style,
            preference_consistency,
            temporal_patterns,
        })
    }

    /// Get ecosystem signature for security validation
    pub fn get_ecosystem_signature(&self) -> &EcosystemSignature {
        &self.machine_ecosystem_signature
    }

    /// Clear ephemeral data for a user (privacy protection)
    pub async fn clear_user_data(&mut self, user_id: Uuid) -> Result<()> {
        info!("Clearing ephemeral data for user {} (privacy protection)", user_id);
        
        {
            let mut observations = self.current_observations.write().await;
            observations.remove(&user_id);
        }
        
        {
            let mut models = self.personality_models.write().await;
            models.remove(&user_id);
        }
        
        {
            let mut contexts = self.environmental_context.write().await;
            contexts.remove(&user_id);
        }

        Ok(())
    }

    /// Get current session statistics (for monitoring)
    pub async fn get_session_stats(&self) -> Result<SessionStats> {
        let observations = self.current_observations.read().await;
        let models = self.personality_models.read().await;
        
        Ok(SessionStats {
            active_users: observations.len(),
            total_interactions: observations.values()
                .map(|obs| obs.query_complexity_history.len())
                .sum(),
            average_session_duration: observations.values()
                .filter_map(|obs| {
                    let duration = Utc::now().signed_duration_since(obs.session_start);
                    Some(duration.num_minutes() as f64)
                })
                .sum::<f64>() / observations.len().max(1) as f64,
            model_confidence: models.values()
                .map(|model| model.behavioral_consistency)
                .sum::<f64>() / models.len().max(1) as f64,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SessionStats {
    pub active_users: usize,
    pub total_interactions: usize,
    pub average_session_duration: f64, // minutes
    pub model_confidence: f64,
}

// Default implementations
impl Default for InteractionPatterns {
    fn default() -> Self {
        Self {
            response_times: Vec::new(),
            query_lengths: Vec::new(),
            interaction_frequency: 0.0,
            session_durations: Vec::new(),
            preferred_interaction_types: HashMap::new(),
        }
    }
}

impl Default for LearningProgression {
    fn default() -> Self {
        Self {
            domains_explored: Vec::new(),
            complexity_progression: HashMap::new(),
            understanding_improvements: HashMap::new(),
            learning_velocity: HashMap::new(),
        }
    }
}

impl Default for TemporalPatterns {
    fn default() -> Self {
        Self {
            active_hours: Vec::new(),
            session_patterns: Vec::new(),
            attention_span_estimates: Vec::new(),
            break_patterns: Vec::new(),
        }
    }
}

impl Default for LearningStyleAssessment {
    fn default() -> Self {
        Self {
            prefers_examples: 0.5,
            prefers_theory_first: 0.5,
            prefers_hands_on: 0.5,
            prefers_visual_aids: 0.5,
            prefers_step_by_step: 0.5,
            learns_from_mistakes: 0.5,
            assessment_confidence: 0.3,
        }
    }
}

impl Default for MotivationIndicators {
    fn default() -> Self {
        Self {
            curiosity_driven: 0.5,
            goal_oriented: 0.5,
            problem_solving_focused: 0.5,
            knowledge_seeking: 0.5,
            practical_application_focused: 0.5,
            social_learning_preference: 0.5,
        }
    }
}

// Implementation for updating from interaction results
impl CurrentObservations {
    fn update_from_interaction_result(
        &mut self,
        _query: &str,
        result: &PersonalizedProcessingResult,
        learning_insights: &[LearningInsight],
    ) -> Result<()> {
        // Update interaction patterns
        self.interaction_patterns.response_times.push(result.processing_time_ms);
        
        // Update learning progression from insights
        for insight in learning_insights {
            if !self.learning_progression.domains_explored.contains(&insight.domain) {
                self.learning_progression.domains_explored.push(insight.domain.clone());
            }
        }
        
        self.last_update = Utc::now();
        Ok(())
    }
}

impl PersonalityModel {
    fn update_from_learning_insights(&mut self, learning_insights: &[LearningInsight]) -> Result<()> {
        for insight in learning_insights {
            // Update confidence in domain understanding
            let current_confidence = self.confidence_levels.get(&insight.domain).unwrap_or(&0.5);
            let confidence_adjustment = match insight.insight_type {
                InsightType::StrengthArea => 0.1,
                InsightType::LearningOpportunity => 0.05,
                InsightType::KnowledgeGap => -0.05,
                InsightType::Misconception => -0.1,
                InsightType::ConceptualConnection => 0.08,
            };
            
            let new_confidence = (current_confidence + confidence_adjustment).max(0.0).min(1.0);
            self.confidence_levels.insert(insight.domain.clone(), new_confidence);
        }
        Ok(())
    }
    
    fn update_confidence_from_result(&mut self, result: &PersonalizedProcessingResult) -> Result<()> {
        // Update behavioral consistency based on processing success
        let success_indicator = result.confidence_score * result.bmd_effectiveness;
        self.behavioral_consistency = (self.behavioral_consistency * 0.9 + success_indicator * 0.1).max(0.0).min(1.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use monkey_tail_kambuzuma::{PersonalizedProcessingResult, ResponseAdaptation, LearningInsight, InsightType, TechnicalDepth, ProcessingMetadata};

    #[tokio::test]
    async fn test_ephemeral_identity_creation() {
        let mut processor = EphemeralIdentityProcessor::new();
        let user_id = Uuid::new_v4();
        
        let interaction_data = InteractionData {
            query_complexity: 0.5,
            domain_context: "test".to_string(),
            user_expertise_level: 0.3,
            communication_preferences: CommunicationPreferences::default(),
            urgency_level: UrgencyLevel::Medium,
            expected_response_type: ResponseType::DirectAnswer,
        };

        let identity = processor.extract_current_identity(user_id, &interaction_data).await.unwrap();
        
        // Should have basic understanding in the test domain
        assert!(identity.understanding_vector.domains.contains_key("test"));
        
        // BMD effectiveness should be in novice range
        let bmd_effectiveness = identity.calculate_bmd_effectiveness("test");
        assert!(bmd_effectiveness >= 0.6 && bmd_effectiveness <= 0.75);
    }

    #[tokio::test]
    async fn test_identity_update_from_interaction() {
        let mut processor = EphemeralIdentityProcessor::new();
        let user_id = Uuid::new_v4();
        
        // Create mock processing result
        let result = PersonalizedProcessingResult {
            content: "Test response".to_string(),
            bmd_effectiveness: 0.8,
            response_adaptation: ResponseAdaptation {
                technical_depth: TechnicalDepth::Intermediate,
                adapted_content: "Adapted response".to_string(),
                interaction_suggestions: vec![],
                follow_up_questions: vec![],
                estimated_comprehension: 0.85,
            },
            learning_insights: vec![
                LearningInsight {
                    domain: "test".to_string(),
                    insight_type: InsightType::StrengthArea,
                    confidence: 0.8,
                    description: "User shows strength in this area".to_string(),
                    suggested_next_steps: vec![],
                }
            ],
            confidence_score: 0.9,
            processing_time_ms: 150,
            metadata: ProcessingMetadata {
                bmd_frames_considered: 3,
                s_entropy_navigation_steps: 5,
                adaptation_iterations: 1,
                domain_expertise_detected: HashMap::new(),
            },
        };

        // Update identity from interaction
        processor.update_identity_from_interaction(
            user_id,
            "test query",
            &result,
            &result.learning_insights,
        ).await.unwrap();

        // Verify the update was processed
        let stats = processor.get_session_stats().await.unwrap();
        assert_eq!(stats.active_users, 1);
        assert_eq!(stats.total_interactions, 1);
    }

    #[tokio::test]
    async fn test_ecosystem_security_validation() {
        let mut processor = EphemeralIdentityProcessor::new();
        let user_id = Uuid::new_v4();
        
        let interaction_data = InteractionData {
            query_complexity: 0.7,
            domain_context: "security_test".to_string(),
            user_expertise_level: 0.6,
            communication_preferences: CommunicationPreferences::default(),
            urgency_level: UrgencyLevel::Medium,
            expected_response_type: ResponseType::Analysis,
        };

        // Should successfully validate ecosystem security
        let identity = processor.extract_current_identity(user_id, &interaction_data).await;
        assert!(identity.is_ok());
    }

    #[tokio::test]
    async fn test_ephemeral_data_clearing() {
        let mut processor = EphemeralIdentityProcessor::new();
        let user_id = Uuid::new_v4();
        
        // Create some data
        let interaction_data = InteractionData {
            query_complexity: 0.5,
            domain_context: "test".to_string(),
            user_expertise_level: 0.5,
            communication_preferences: CommunicationPreferences::default(),
            urgency_level: UrgencyLevel::Medium,
            expected_response_type: ResponseType::DirectAnswer,
        };

        processor.extract_current_identity(user_id, &interaction_data).await.unwrap();
        
        // Verify data exists
        let stats_before = processor.get_session_stats().await.unwrap();
        assert_eq!(stats_before.active_users, 1);
        
        // Clear data
        processor.clear_user_data(user_id).await.unwrap();
        
        // Verify data is cleared
        let stats_after = processor.get_session_stats().await.unwrap();
        assert_eq!(stats_after.active_users, 0);
    }
}
