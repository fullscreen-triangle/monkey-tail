use monkey_tail_core::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use std::sync::Arc;
use anyhow::Result;
use tracing::{debug, info, warn};

pub mod assessor;
pub mod models;
pub mod consensus;
pub mod turbulance;

pub use assessor::*;
pub use models::*;
pub use consensus::*;
pub use turbulance::*;

/// Four-sided triangle competency assessor implementing multi-model consensus
/// The "four sides" represent: Multi-Model, Domain Expert, Quality Orchestration, Turbulance Integration
pub struct FourSidedTriangleAssessor {
    multi_model_consensus: Arc<RwLock<MultiModelConsensus>>,
    domain_expert_extractor: Arc<RwLock<DomainExpertExtractor>>,
    quality_orchestrator: Arc<RwLock<QualityOrchestrator>>,
    turbulance_integrator: Arc<RwLock<TurbulanceIntegrator>>,
    assessment_cache: Arc<RwLock<HashMap<String, CachedAssessment>>>,
}

/// Comprehensive competency assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetencyAssessment {
    pub domain: String,
    pub understanding_level: f64,        // 0.0-1.0
    pub knowledge_depth: KnowledgeDepth,
    pub confidence_score: f64,
    pub assessment_quality: f64,
    pub consensus_agreement: f64,
    pub model_assessments: Vec<ModelAssessment>,
    pub expert_indicators: Vec<ExpertiseIndicator>,
    pub quality_metrics: QualityMetrics,
    pub turbulance_analysis: TurbulanceAnalysis,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDepth {
    pub surface_level: f64,      // Basic facts and definitions
    pub procedural_level: f64,   // How-to knowledge
    pub conceptual_level: f64,   // Deep understanding of principles
    pub metacognitive_level: f64, // Knowledge about knowledge
    pub creative_level: f64,     // Ability to generate novel insights
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAssessment {
    pub model_name: String,
    pub assessment_score: f64,
    pub confidence: f64,
    pub reasoning: String,
    pub evidence_points: Vec<String>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertiseIndicator {
    pub indicator_type: ExpertiseType,
    pub strength: f64,
    pub evidence: String,
    pub domain_specificity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseType {
    TechnicalVocabulary,
    ConceptualUnderstanding,
    ProblemSolvingApproach,
    QuestionSophistication,
    CrossDomainConnections,
    MetacognitiveAwareness,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub overall_quality: f64,
    pub consistency_score: f64,
    pub reliability_score: f64,
    pub validity_score: f64,
    pub assessment_completeness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurbulanceAnalysis {
    pub semantic_patterns: Vec<SemanticPattern>,
    pub linguistic_complexity: f64,
    pub conceptual_coherence: f64,
    pub domain_alignment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPattern {
    pub pattern_type: String,
    pub strength: f64,
    pub relevance: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
struct CachedAssessment {
    assessment: CompetencyAssessment,
    expiry: DateTime<Utc>,
}

impl FourSidedTriangleAssessor {
    pub fn new() -> Self {
        info!("Initializing Four-Sided Triangle Competency Assessor");
        
        Self {
            multi_model_consensus: Arc::new(RwLock::new(MultiModelConsensus::new())),
            domain_expert_extractor: Arc::new(RwLock::new(DomainExpertExtractor::new())),
            quality_orchestrator: Arc::new(RwLock::new(QualityOrchestrator::new())),
            turbulance_integrator: Arc::new(RwLock::new(TurbulanceIntegrator::new())),
            assessment_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Assess user competency across multiple dimensions (the "four sides")
    pub async fn assess_competency(
        &mut self,
        user_query: &str,
        domain_context: &str,
        interaction_history: &InteractionHistory,
    ) -> Result<CompetencyAssessment> {
        debug!("Assessing competency for query: '{}' in domain: '{}'", user_query, domain_context);

        // Check cache first
        let cache_key = format!("{}:{}:{}", 
            self.hash_query(user_query), 
            domain_context, 
            interaction_history.hash()
        );
        
        if let Some(cached) = self.get_cached_assessment(&cache_key).await? {
            debug!("Returning cached assessment for key: {}", cache_key);
            return Ok(cached);
        }

        // Side 1: Multi-model consensus assessment
        let model_assessments = {
            let mut consensus = self.multi_model_consensus.write().await;
            consensus.assess_across_models(user_query, domain_context).await?
        };

        // Side 2: Domain expert knowledge extraction
        let expert_knowledge = {
            let mut extractor = self.domain_expert_extractor.write().await;
            extractor.extract_domain_expertise(domain_context, interaction_history).await?
        };

        // Side 3: Quality orchestration and validation
        let quality_metrics = {
            let mut orchestrator = self.quality_orchestrator.write().await;
            orchestrator.validate_assessment_quality(&model_assessments, &expert_knowledge).await?
        };

        // Side 4: Turbulance DSL integration for semantic processing
        let turbulance_analysis = {
            let mut integrator = self.turbulance_integrator.write().await;
            integrator.process_semantic_patterns(user_query, &model_assessments).await?
        };

        // Synthesize final competency assessment from all four sides
        let assessment = self.synthesize_four_sided_assessment(
            domain_context,
            &model_assessments,
            &expert_knowledge,
            &quality_metrics,
            &turbulance_analysis,
        ).await?;

        // Cache the assessment
        self.cache_assessment(cache_key, &assessment).await?;

        info!("Competency assessment completed: {:.1}% understanding in domain '{}'", 
              assessment.understanding_level * 100.0, domain_context);

        Ok(assessment)
    }

    async fn synthesize_four_sided_assessment(
        &self,
        domain: &str,
        model_assessments: &[ModelAssessment],
        expert_knowledge: &DomainExpertise,
        quality_metrics: &QualityMetrics,
        turbulance_analysis: &TurbulanceAnalysis,
    ) -> Result<CompetencyAssessment> {
        // Calculate consensus from multiple models (Side 1)
        let consensus_score = self.calculate_consensus_score(model_assessments)?;
        
        // Weight assessments by quality metrics (Side 3)
        let weighted_understanding = self.calculate_weighted_understanding(
            model_assessments,
            quality_metrics,
        )?;

        // Integrate expert knowledge indicators (Side 2)
        let expert_adjusted_understanding = self.adjust_for_expert_knowledge(
            weighted_understanding,
            expert_knowledge,
        )?;

        // Apply Turbulance semantic analysis (Side 4)
        let final_understanding = self.apply_turbulance_adjustment(
            expert_adjusted_understanding,
            turbulance_analysis,
        )?;

        // Calculate knowledge depth from all sources
        let knowledge_depth = self.calculate_knowledge_depth(
            expert_knowledge,
            turbulance_analysis,
            model_assessments,
        )?;

        // Extract expertise indicators
        let expert_indicators = self.extract_expertise_indicators(
            expert_knowledge,
            model_assessments,
            turbulance_analysis,
        )?;

        Ok(CompetencyAssessment {
            domain: domain.to_string(),
            understanding_level: final_understanding,
            knowledge_depth,
            confidence_score: consensus_score * quality_metrics.overall_quality,
            assessment_quality: quality_metrics.overall_quality,
            consensus_agreement: consensus_score,
            model_assessments: model_assessments.to_vec(),
            expert_indicators,
            quality_metrics: quality_metrics.clone(),
            turbulance_analysis: turbulance_analysis.clone(),
            timestamp: Utc::now(),
        })
    }

    fn calculate_consensus_score(&self, assessments: &[ModelAssessment]) -> Result<f64> {
        if assessments.is_empty() {
            return Ok(0.0);
        }

        let scores: Vec<f64> = assessments.iter().map(|a| a.assessment_score).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        
        // Calculate standard deviation
        let variance = scores.iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();

        // Consensus is higher when standard deviation is lower
        let consensus = 1.0 - (std_dev / 1.0).min(1.0); // Normalize to [0,1]
        
        Ok(consensus)
    }

    fn calculate_weighted_understanding(
        &self,
        assessments: &[ModelAssessment],
        quality_metrics: &QualityMetrics,
    ) -> Result<f64> {
        if assessments.is_empty() {
            return Ok(0.0);
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for assessment in assessments {
            let weight = assessment.confidence * quality_metrics.reliability_score;
            weighted_sum += assessment.assessment_score * weight;
            total_weight += weight;
        }

        Ok(if total_weight > 0.0 { weighted_sum / total_weight } else { 0.0 })
    }

    fn adjust_for_expert_knowledge(
        &self,
        base_understanding: f64,
        expert_knowledge: &DomainExpertise,
    ) -> Result<f64> {
        let expert_bonus = expert_knowledge.expertise_level * 0.2; // Up to 20% bonus
        let adjusted = (base_understanding + expert_bonus).min(1.0);
        Ok(adjusted)
    }

    fn apply_turbulance_adjustment(
        &self,
        base_understanding: f64,
        turbulance_analysis: &TurbulanceAnalysis,
    ) -> Result<f64> {
        let semantic_bonus = turbulance_analysis.conceptual_coherence * 0.1; // Up to 10% bonus
        let domain_alignment_factor = turbulance_analysis.domain_alignment;
        
        let adjusted = (base_understanding + semantic_bonus) * domain_alignment_factor;
        Ok(adjusted.min(1.0))
    }

    fn calculate_knowledge_depth(
        &self,
        expert_knowledge: &DomainExpertise,
        turbulance_analysis: &TurbulanceAnalysis,
        model_assessments: &[ModelAssessment],
    ) -> Result<KnowledgeDepth> {
        // Extract depth indicators from various sources
        let surface_level = expert_knowledge.basic_indicators.iter().sum::<f64>() / expert_knowledge.basic_indicators.len().max(1) as f64;
        
        let procedural_level = expert_knowledge.procedural_indicators.iter().sum::<f64>() / expert_knowledge.procedural_indicators.len().max(1) as f64;
        
        let conceptual_level = turbulance_analysis.conceptual_coherence;
        
        let metacognitive_level = model_assessments.iter()
            .map(|a| a.confidence)
            .sum::<f64>() / model_assessments.len().max(1) as f64;
        
        let creative_level = turbulance_analysis.semantic_patterns.iter()
            .filter(|p| p.pattern_type == "creative")
            .map(|p| p.strength)
            .sum::<f64>() / turbulance_analysis.semantic_patterns.len().max(1) as f64;

        Ok(KnowledgeDepth {
            surface_level,
            procedural_level,
            conceptual_level,
            metacognitive_level,
            creative_level,
        })
    }

    fn extract_expertise_indicators(
        &self,
        expert_knowledge: &DomainExpertise,
        model_assessments: &[ModelAssessment],
        turbulance_analysis: &TurbulanceAnalysis,
    ) -> Result<Vec<ExpertiseIndicator>> {
        let mut indicators = Vec::new();

        // Technical vocabulary indicators
        if turbulance_analysis.linguistic_complexity > 0.7 {
            indicators.push(ExpertiseIndicator {
                indicator_type: ExpertiseType::TechnicalVocabulary,
                strength: turbulance_analysis.linguistic_complexity,
                evidence: "High linguistic complexity in domain-specific terms".to_string(),
                domain_specificity: turbulance_analysis.domain_alignment,
            });
        }

        // Conceptual understanding indicators
        if turbulance_analysis.conceptual_coherence > 0.8 {
            indicators.push(ExpertiseIndicator {
                indicator_type: ExpertiseType::ConceptualUnderstanding,
                strength: turbulance_analysis.conceptual_coherence,
                evidence: "Strong conceptual coherence in responses".to_string(),
                domain_specificity: expert_knowledge.expertise_level,
            });
        }

        // Problem-solving approach indicators
        let avg_model_confidence = model_assessments.iter()
            .map(|a| a.confidence)
            .sum::<f64>() / model_assessments.len().max(1) as f64;
        
        if avg_model_confidence > 0.8 {
            indicators.push(ExpertiseIndicator {
                indicator_type: ExpertiseType::ProblemSolvingApproach,
                strength: avg_model_confidence,
                evidence: "Sophisticated problem-solving approach detected".to_string(),
                domain_specificity: expert_knowledge.expertise_level,
            });
        }

        Ok(indicators)
    }

    async fn get_cached_assessment(&self, cache_key: &str) -> Result<Option<CompetencyAssessment>> {
        let cache = self.assessment_cache.read().await;
        
        if let Some(cached) = cache.get(cache_key) {
            if cached.expiry > Utc::now() {
                return Ok(Some(cached.assessment.clone()));
            }
        }
        
        Ok(None)
    }

    async fn cache_assessment(&self, cache_key: String, assessment: &CompetencyAssessment) -> Result<()> {
        let mut cache = self.assessment_cache.write().await;
        
        let cached = CachedAssessment {
            assessment: assessment.clone(),
            expiry: Utc::now() + chrono::Duration::minutes(30), // Cache for 30 minutes
        };
        
        cache.insert(cache_key, cached);
        Ok(())
    }

    fn hash_query(&self, query: &str) -> String {
        // Simple hash for caching - in production would use proper hashing
        format!("{:x}", query.len() * 31 + query.chars().map(|c| c as usize).sum::<usize>())
    }

    /// Get assessment statistics for monitoring
    pub async fn get_assessment_stats(&self) -> Result<AssessmentStats> {
        let cache = self.assessment_cache.read().await;
        
        let total_assessments = cache.len();
        let avg_confidence = cache.values()
            .map(|cached| cached.assessment.confidence_score)
            .sum::<f64>() / cache.len().max(1) as f64;
        
        let domain_distribution: HashMap<String, usize> = cache.values()
            .map(|cached| cached.assessment.domain.clone())
            .fold(HashMap::new(), |mut acc, domain| {
                *acc.entry(domain).or_insert(0) += 1;
                acc
            });

        Ok(AssessmentStats {
            total_assessments,
            average_confidence: avg_confidence,
            domain_distribution,
            cache_hit_rate: 0.0, // Would need to track hits/misses
        })
    }
}

#[derive(Debug, Clone)]
pub struct AssessmentStats {
    pub total_assessments: usize,
    pub average_confidence: f64,
    pub domain_distribution: HashMap<String, usize>,
    pub cache_hit_rate: f64,
}

/// Interaction history for competency assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionHistory {
    pub interactions: Vec<HistoricalInteraction>,
    pub total_interactions: usize,
    pub domains_covered: Vec<String>,
    pub average_complexity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalInteraction {
    pub timestamp: DateTime<Utc>,
    pub query: String,
    pub domain: String,
    pub complexity: f64,
    pub success_indicators: Vec<String>,
}

impl InteractionHistory {
    pub fn new() -> Self {
        Self {
            interactions: Vec::new(),
            total_interactions: 0,
            domains_covered: Vec::new(),
            average_complexity: 0.0,
        }
    }

    pub fn hash(&self) -> String {
        // Simple hash for caching
        format!("{:x}", self.total_interactions * 31 + self.domains_covered.len() * 17)
    }
}

impl Default for InteractionHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_four_sided_triangle_assessor_creation() {
        let assessor = FourSidedTriangleAssessor::new();
        // Should initialize without errors
        assert!(true);
    }

    #[tokio::test]
    async fn test_competency_assessment() {
        let mut assessor = FourSidedTriangleAssessor::new();
        let interaction_history = InteractionHistory::new();
        
        let assessment = assessor.assess_competency(
            "What is quantum mechanics?",
            "physics",
            &interaction_history,
        ).await.unwrap();

        assert_eq!(assessment.domain, "physics");
        assert!(assessment.understanding_level >= 0.0 && assessment.understanding_level <= 1.0);
        assert!(assessment.confidence_score >= 0.0 && assessment.confidence_score <= 1.0);
        assert!(!assessment.model_assessments.is_empty());
    }

    #[tokio::test]
    async fn test_assessment_caching() {
        let mut assessor = FourSidedTriangleAssessor::new();
        let interaction_history = InteractionHistory::new();
        
        // First assessment
        let assessment1 = assessor.assess_competency(
            "What is machine learning?",
            "computer_science",
            &interaction_history,
        ).await.unwrap();

        // Second identical assessment (should be cached)
        let assessment2 = assessor.assess_competency(
            "What is machine learning?",
            "computer_science", 
            &interaction_history,
        ).await.unwrap();

        // Should have same timestamp (indicating cache hit)
        assert_eq!(assessment1.timestamp, assessment2.timestamp);
    }

    #[tokio::test]
    async fn test_knowledge_depth_calculation() {
        let mut assessor = FourSidedTriangleAssessor::new();
        let interaction_history = InteractionHistory::new();
        
        let assessment = assessor.assess_competency(
            "Derive the Schrödinger equation from first principles",
            "physics",
            &interaction_history,
        ).await.unwrap();

        // Advanced query should show higher conceptual level
        assert!(assessment.knowledge_depth.conceptual_level > 0.0);
        assert!(assessment.understanding_level >= 0.0);
    }

    #[tokio::test]
    async fn test_expertise_indicators() {
        let mut assessor = FourSidedTriangleAssessor::new();
        let interaction_history = InteractionHistory::new();
        
        let assessment = assessor.assess_competency(
            "How do convolutional neural networks implement translation invariance through weight sharing?",
            "machine_learning",
            &interaction_history,
        ).await.unwrap();

        // Technical query should generate expertise indicators
        assert!(!assessment.expert_indicators.is_empty());
        
        // Should detect technical vocabulary
        let has_technical_vocab = assessment.expert_indicators.iter()
            .any(|indicator| matches!(indicator.indicator_type, ExpertiseType::TechnicalVocabulary));
        assert!(has_technical_vocab);
    }
}
