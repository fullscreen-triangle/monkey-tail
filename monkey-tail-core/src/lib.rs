use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub mod errors;
pub mod types;

pub use errors::*;
pub use types::*;

/// Core semantic identity representing what the AI understands about the user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticIdentity {
    pub understanding_vector: UnderstandingVector,
    pub knowledge_depth_matrix: KnowledgeDepthMatrix,
    pub motivation_mapping: MotivationMapping,
    pub communication_patterns: CommunicationPatterns,
    pub temporal_context: TemporalContext,
    pub emotional_state: EmotionalStateVector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandingVector {
    pub domains: HashMap<String, f64>, // Domain -> comprehension level (0.0-1.0)
    pub cross_domain_connections: Vec<(String, String, f64)>, // (domain1, domain2, connection_strength)
    pub learning_velocity: HashMap<String, f64>, // Rate of understanding change
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDepthMatrix {
    pub surface_knowledge: HashMap<String, f64>,      // Basic facts and definitions
    pub procedural_knowledge: HashMap<String, f64>,   // How to do things
    pub conceptual_knowledge: HashMap<String, f64>,   // Deep understanding
    pub metacognitive_knowledge: HashMap<String, f64>, // Knowledge about knowledge
    pub revolutionary_insights: HashMap<String, f64>,  // Paradigm-shifting understanding
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MotivationMapping {
    pub intrinsic_motivators: HashMap<String, f64>,
    pub extrinsic_motivators: HashMap<String, f64>,
    pub goal_hierarchies: Vec<Goal>,
    pub value_system: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: Uuid,
    pub description: String,
    pub priority: f64,
    pub progress: f64,
    pub deadline: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPatterns {
    pub preferred_detail_level: DetailLevel,
    pub communication_style: CommunicationStyle,
    pub learning_style: LearningStyle,
    pub interaction_preferences: InteractionPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetailLevel {
    Overview,      // High-level summaries
    Moderate,      // Balanced detail
    Comprehensive, // Thorough explanations
    Expert,        // Technical depth
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStyle {
    Direct,      // Straightforward, no fluff
    Detailed,    // Comprehensive explanations
    Interactive, // Question-based dialogue
    Technical,   // Formal, precise language
    Creative,    // Analogies and metaphors
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearningStyle {
    pub visual_preference: f64,
    pub auditory_preference: f64,
    pub kinesthetic_preference: f64,
    pub reading_preference: f64,
    pub example_driven: f64,
    pub theory_first: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InteractionPreferences {
    pub response_speed: ResponseSpeed,
    pub feedback_frequency: FeedbackFrequency,
    pub challenge_level: f64, // 0.0 = easy, 1.0 = maximum challenge
    pub exploration_vs_exploitation: f64, // 0.0 = stick to known, 1.0 = explore new
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseSpeed {
    Immediate,  // Real-time responses
    Thoughtful, // Take time for quality
    Adaptive,   // Match user's pace
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackFrequency {
    Continuous, // Constant feedback
    Periodic,   // Regular intervals
    OnDemand,   // Only when requested
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemporalContext {
    pub current_session_start: Option<DateTime<Utc>>,
    pub interaction_history_summary: InteractionSummary,
    pub time_of_day_patterns: HashMap<u8, f64>, // Hour -> activity level
    pub seasonal_patterns: HashMap<String, f64>,
    pub attention_span_estimate: f64, // Minutes
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InteractionSummary {
    pub total_interactions: u64,
    pub average_session_length: f64, // Minutes
    pub most_active_domains: Vec<String>,
    pub learning_progression: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalStateVector {
    pub current_mood: HashMap<String, f64>, // Emotion -> intensity
    pub stress_level: f64,
    pub engagement_level: f64,
    pub confidence_level: f64,
    pub curiosity_level: f64,
}

impl Default for CommunicationPatterns {
    fn default() -> Self {
        Self {
            preferred_detail_level: DetailLevel::Moderate,
            communication_style: CommunicationStyle::Direct,
            learning_style: LearningStyle::default(),
            interaction_preferences: InteractionPreferences::default(),
        }
    }
}

impl Default for ResponseSpeed {
    fn default() -> Self {
        ResponseSpeed::Adaptive
    }
}

impl Default for FeedbackFrequency {
    fn default() -> Self {
        FeedbackFrequency::Periodic
    }
}

impl SemanticIdentity {
    pub fn new() -> Self {
        Self {
            understanding_vector: UnderstandingVector {
                domains: HashMap::new(),
                cross_domain_connections: Vec::new(),
                learning_velocity: HashMap::new(),
            },
            knowledge_depth_matrix: KnowledgeDepthMatrix {
                surface_knowledge: HashMap::new(),
                procedural_knowledge: HashMap::new(),
                conceptual_knowledge: HashMap::new(),
                metacognitive_knowledge: HashMap::new(),
                revolutionary_insights: HashMap::new(),
            },
            motivation_mapping: MotivationMapping::default(),
            communication_patterns: CommunicationPatterns::default(),
            temporal_context: TemporalContext::default(),
            emotional_state: EmotionalStateVector::default(),
        }
    }

    /// Calculate BMD (Biological Maxwell Demon) effectiveness based on user understanding
    /// Returns effectiveness score from 0.6 (novice) to 0.95 (expert)
    pub fn calculate_bmd_effectiveness(&self, domain: &str) -> f64 {
        let base_effectiveness = 0.6; // Minimum 60% for novice users

        let understanding_bonus = self.understanding_vector.domains
            .get(domain)
            .unwrap_or(&0.0) * 0.35; // Up to 35% bonus for understanding

        let depth_bonus = self.calculate_depth_bonus(domain) * 0.05; // Up to 5% for depth

        (base_effectiveness + understanding_bonus + depth_bonus).min(0.95) // Max 95% effectiveness
    }

    fn calculate_depth_bonus(&self, domain: &str) -> f64 {
        let surface = self.knowledge_depth_matrix.surface_knowledge.get(domain).unwrap_or(&0.0);
        let procedural = self.knowledge_depth_matrix.procedural_knowledge.get(domain).unwrap_or(&0.0);
        let conceptual = self.knowledge_depth_matrix.conceptual_knowledge.get(domain).unwrap_or(&0.0);
        let metacognitive = self.knowledge_depth_matrix.metacognitive_knowledge.get(domain).unwrap_or(&0.0);
        let revolutionary = self.knowledge_depth_matrix.revolutionary_insights.get(domain).unwrap_or(&0.0);

        // Weighted average favoring deeper knowledge
        (surface * 0.1 + procedural * 0.2 + conceptual * 0.3 + metacognitive * 0.2 + revolutionary * 0.2)
    }

    /// Update understanding in a domain based on interaction success
    pub fn update_domain_understanding(&mut self, domain: &str, interaction_success: f64, complexity: f64) {
        let current_understanding = self.understanding_vector.domains
            .get(domain)
            .unwrap_or(&0.0);

        // Learning rate based on current understanding (slower as expertise increases)
        let learning_rate = (1.0 - current_understanding) * 0.1;
        
        // Adjust based on interaction success and complexity
        let understanding_delta = learning_rate * interaction_success * complexity;
        
        let new_understanding = (current_understanding + understanding_delta).min(1.0).max(0.0);
        
        self.understanding_vector.domains.insert(domain.to_string(), new_understanding);
        
        // Update learning velocity
        let velocity = understanding_delta / learning_rate.max(0.001);
        self.understanding_vector.learning_velocity.insert(domain.to_string(), velocity);
    }

    /// Get the user's expertise level in a domain
    pub fn get_expertise_level(&self, domain: &str) -> ExpertiseLevel {
        let understanding = self.understanding_vector.domains.get(domain).unwrap_or(&0.0);
        
        match *understanding {
            x if x < 0.2 => ExpertiseLevel::Novice,
            x if x < 0.5 => ExpertiseLevel::Beginner,
            x if x < 0.7 => ExpertiseLevel::Intermediate,
            x if x < 0.9 => ExpertiseLevel::Advanced,
            _ => ExpertiseLevel::Expert,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExpertiseLevel {
    Novice,
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

impl Default for SemanticIdentity {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bmd_effectiveness_scaling() {
        let mut identity = SemanticIdentity::new();
        
        // Test novice level (should be around 60%)
        let novice_effectiveness = identity.calculate_bmd_effectiveness("unknown_domain");
        assert!(novice_effectiveness >= 0.6 && novice_effectiveness <= 0.65);
        
        // Add expert-level understanding
        identity.understanding_vector.domains.insert("physics".to_string(), 0.9);
        identity.knowledge_depth_matrix.conceptual_knowledge.insert("physics".to_string(), 0.8);
        
        // Test expert level (should be around 90-95%)
        let expert_effectiveness = identity.calculate_bmd_effectiveness("physics");
        assert!(expert_effectiveness >= 0.9 && expert_effectiveness <= 0.95);
    }

    #[test]
    fn test_domain_understanding_update() {
        let mut identity = SemanticIdentity::new();
        
        // Initial understanding should be 0
        assert_eq!(identity.understanding_vector.domains.get("rust"), None);
        
        // Simulate successful interaction with moderate complexity
        identity.update_domain_understanding("rust", 0.8, 0.6);
        
        let understanding = identity.understanding_vector.domains.get("rust").unwrap();
        assert!(*understanding > 0.0);
        assert!(*understanding < 0.1); // Should be small for first interaction
        
        // Simulate multiple successful interactions
        for _ in 0..10 {
            identity.update_domain_understanding("rust", 0.9, 0.7);
        }
        
        let final_understanding = identity.understanding_vector.domains.get("rust").unwrap();
        assert!(*final_understanding > *understanding); // Should have improved
    }

    #[test]
    fn test_expertise_level_classification() {
        let mut identity = SemanticIdentity::new();
        
        // Test novice level
        identity.understanding_vector.domains.insert("domain1".to_string(), 0.1);
        assert_eq!(identity.get_expertise_level("domain1"), ExpertiseLevel::Novice);
        
        // Test intermediate level
        identity.understanding_vector.domains.insert("domain2".to_string(), 0.6);
        assert_eq!(identity.get_expertise_level("domain2"), ExpertiseLevel::Intermediate);
        
        // Test expert level
        identity.understanding_vector.domains.insert("domain3".to_string(), 0.95);
        assert_eq!(identity.get_expertise_level("domain3"), ExpertiseLevel::Expert);
    }
}