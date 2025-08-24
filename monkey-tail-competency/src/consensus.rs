use crate::*;
use anyhow::Result;
use tracing::debug;

/// Domain expert knowledge extractor (Side 2 of the four-sided triangle)
pub struct DomainExpertExtractor {
    expert_knowledge_base: HashMap<String, DomainKnowledgeBase>,
    expertise_patterns: ExpertisePatterns,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainExpertise {
    pub domain: String,
    pub expertise_level: f64,
    pub basic_indicators: Vec<f64>,
    pub procedural_indicators: Vec<f64>,
    pub conceptual_indicators: Vec<f64>,
    pub expert_patterns: Vec<ExpertPattern>,
    pub knowledge_gaps: Vec<String>,
    pub strength_areas: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DomainKnowledgeBase {
    pub domain_name: String,
    pub core_concepts: Vec<String>,
    pub advanced_concepts: Vec<String>,
    pub expert_vocabulary: Vec<String>,
    pub common_misconceptions: Vec<String>,
    pub expertise_indicators: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ExpertisePatterns {
    pub novice_patterns: Vec<String>,
    pub intermediate_patterns: Vec<String>,
    pub advanced_patterns: Vec<String>,
    pub expert_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPattern {
    pub pattern_type: String,
    pub confidence: f64,
    pub evidence: String,
    pub level: ExpertiseLevel,
}

impl DomainExpertExtractor {
    pub fn new() -> Self {
        let mut extractor = Self {
            expert_knowledge_base: HashMap::new(),
            expertise_patterns: ExpertisePatterns::new(),
        };
        
        extractor.initialize_knowledge_bases();
        extractor
    }

    pub async fn extract_domain_expertise(
        &mut self,
        domain: &str,
        interaction_history: &InteractionHistory,
    ) -> Result<DomainExpertise> {
        debug!("Extracting domain expertise for: {}", domain);

        let knowledge_base = self.get_or_create_knowledge_base(domain);
        
        // Analyze interaction history for expertise indicators
        let basic_indicators = self.analyze_basic_indicators(interaction_history, &knowledge_base)?;
        let procedural_indicators = self.analyze_procedural_indicators(interaction_history, &knowledge_base)?;
        let conceptual_indicators = self.analyze_conceptual_indicators(interaction_history, &knowledge_base)?;
        
        // Extract expert patterns
        let expert_patterns = self.extract_expert_patterns(interaction_history, &knowledge_base)?;
        
        // Calculate overall expertise level
        let expertise_level = self.calculate_expertise_level(
            &basic_indicators,
            &procedural_indicators,
            &conceptual_indicators,
            &expert_patterns,
        )?;

        // Identify knowledge gaps and strengths
        let (knowledge_gaps, strength_areas) = self.identify_gaps_and_strengths(
            interaction_history,
            &knowledge_base,
            expertise_level,
        )?;

        Ok(DomainExpertise {
            domain: domain.to_string(),
            expertise_level,
            basic_indicators,
            procedural_indicators,
            conceptual_indicators,
            expert_patterns,
            knowledge_gaps,
            strength_areas,
        })
    }

    fn analyze_basic_indicators(
        &self,
        history: &InteractionHistory,
        knowledge_base: &DomainKnowledgeBase,
    ) -> Result<Vec<f64>> {
        let mut indicators = Vec::new();
        
        for interaction in &history.interactions {
            let query_lower = interaction.query.to_lowercase();
            
            // Check for basic concept usage
            let basic_concept_usage = knowledge_base.core_concepts.iter()
                .filter(|&concept| query_lower.contains(&concept.to_lowercase()))
                .count() as f64 / knowledge_base.core_concepts.len() as f64;
            
            // Check for basic vocabulary
            let vocabulary_usage = knowledge_base.expert_vocabulary.iter()
                .take(10) // First 10 are considered basic
                .filter(|&vocab| query_lower.contains(&vocab.to_lowercase()))
                .count() as f64 / 10.0;
            
            // Basic question patterns
            let basic_patterns = self.expertise_patterns.novice_patterns.iter()
                .filter(|&pattern| query_lower.contains(pattern))
                .count() as f64 / self.expertise_patterns.novice_patterns.len() as f64;
            
            let indicator_score = (basic_concept_usage + vocabulary_usage + basic_patterns) / 3.0;
            indicators.push(indicator_score);
        }
        
        Ok(indicators)
    }

    fn analyze_procedural_indicators(
        &self,
        history: &InteractionHistory,
        knowledge_base: &DomainKnowledgeBase,
    ) -> Result<Vec<f64>> {
        let mut indicators = Vec::new();
        
        for interaction in &history.interactions {
            let query_lower = interaction.query.to_lowercase();
            
            // Check for procedural language
            let procedural_words = ["how to", "steps", "process", "method", "procedure", "algorithm"];
            let procedural_usage = procedural_words.iter()
                .filter(|&word| query_lower.contains(word))
                .count() as f64 / procedural_words.len() as f64;
            
            // Check for implementation questions
            let implementation_words = ["implement", "build", "create", "develop", "design"];
            let implementation_usage = implementation_words.iter()
                .filter(|&word| query_lower.contains(word))
                .count() as f64 / implementation_words.len() as f64;
            
            // Intermediate patterns
            let intermediate_patterns = self.expertise_patterns.intermediate_patterns.iter()
                .filter(|&pattern| query_lower.contains(pattern))
                .count() as f64 / self.expertise_patterns.intermediate_patterns.len() as f64;
            
            let indicator_score = (procedural_usage + implementation_usage + intermediate_patterns) / 3.0;
            indicators.push(indicator_score);
        }
        
        Ok(indicators)
    }

    fn analyze_conceptual_indicators(
        &self,
        history: &InteractionHistory,
        knowledge_base: &DomainKnowledgeBase,
    ) -> Result<Vec<f64>> {
        let mut indicators = Vec::new();
        
        for interaction in &history.interactions {
            let query_lower = interaction.query.to_lowercase();
            
            // Check for advanced concept usage
            let advanced_concept_usage = knowledge_base.advanced_concepts.iter()
                .filter(|&concept| query_lower.contains(&concept.to_lowercase()))
                .count() as f64 / knowledge_base.advanced_concepts.len().max(1) as f64;
            
            // Check for theoretical language
            let theoretical_words = ["theory", "principle", "framework", "model", "paradigm"];
            let theoretical_usage = theoretical_words.iter()
                .filter(|&word| query_lower.contains(word))
                .count() as f64 / theoretical_words.len() as f64;
            
            // Check for analytical language
            let analytical_words = ["analyze", "evaluate", "compare", "contrast", "synthesize"];
            let analytical_usage = analytical_words.iter()
                .filter(|&word| query_lower.contains(word))
                .count() as f64 / analytical_words.len() as f64;
            
            // Advanced patterns
            let advanced_patterns = self.expertise_patterns.advanced_patterns.iter()
                .filter(|&pattern| query_lower.contains(pattern))
                .count() as f64 / self.expertise_patterns.advanced_patterns.len() as f64;
            
            let indicator_score = (advanced_concept_usage + theoretical_usage + analytical_usage + advanced_patterns) / 4.0;
            indicators.push(indicator_score);
        }
        
        Ok(indicators)
    }

    fn extract_expert_patterns(
        &self,
        history: &InteractionHistory,
        knowledge_base: &DomainKnowledgeBase,
    ) -> Result<Vec<ExpertPattern>> {
        let mut patterns = Vec::new();
        
        for interaction in &history.interactions {
            let query_lower = interaction.query.to_lowercase();
            
            // Check for expert vocabulary usage
            let expert_vocab_count = knowledge_base.expert_vocabulary.iter()
                .filter(|&vocab| query_lower.contains(&vocab.to_lowercase()))
                .count();
            
            if expert_vocab_count > 2 {
                patterns.push(ExpertPattern {
                    pattern_type: "expert_vocabulary".to_string(),
                    confidence: (expert_vocab_count as f64 / 5.0).min(1.0),
                    evidence: format!("Uses {} expert terms", expert_vocab_count),
                    level: ExpertiseLevel::Advanced,
                });
            }
            
            // Check for expert question patterns
            let expert_pattern_matches = self.expertise_patterns.expert_patterns.iter()
                .filter(|&pattern| query_lower.contains(pattern))
                .count();
            
            if expert_pattern_matches > 0 {
                patterns.push(ExpertPattern {
                    pattern_type: "expert_questioning".to_string(),
                    confidence: (expert_pattern_matches as f64 / 3.0).min(1.0),
                    evidence: format!("Uses {} expert question patterns", expert_pattern_matches),
                    level: ExpertiseLevel::Expert,
                });
            }
            
            // Check for misconception avoidance
            let misconception_mentions = knowledge_base.common_misconceptions.iter()
                .filter(|&misconception| query_lower.contains(&misconception.to_lowercase()))
                .count();
            
            if misconception_mentions == 0 && interaction.query.len() > 50 {
                patterns.push(ExpertPattern {
                    pattern_type: "misconception_avoidance".to_string(),
                    confidence: 0.7,
                    evidence: "Avoids common misconceptions in complex query".to_string(),
                    level: ExpertiseLevel::Intermediate,
                });
            }
        }
        
        Ok(patterns)
    }

    fn calculate_expertise_level(
        &self,
        basic_indicators: &[f64],
        procedural_indicators: &[f64],
        conceptual_indicators: &[f64],
        expert_patterns: &[ExpertPattern],
    ) -> Result<f64> {
        let basic_avg = basic_indicators.iter().sum::<f64>() / basic_indicators.len().max(1) as f64;
        let procedural_avg = procedural_indicators.iter().sum::<f64>() / procedural_indicators.len().max(1) as f64;
        let conceptual_avg = conceptual_indicators.iter().sum::<f64>() / conceptual_indicators.len().max(1) as f64;
        
        let expert_pattern_score = expert_patterns.iter()
            .map(|p| p.confidence)
            .sum::<f64>() / expert_patterns.len().max(1) as f64;
        
        // Weighted combination favoring higher-level indicators
        let expertise_level = (
            basic_avg * 0.2 +
            procedural_avg * 0.3 +
            conceptual_avg * 0.4 +
            expert_pattern_score * 0.1
        ).min(1.0);
        
        Ok(expertise_level)
    }

    fn identify_gaps_and_strengths(
        &self,
        history: &InteractionHistory,
        knowledge_base: &DomainKnowledgeBase,
        expertise_level: f64,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let mut knowledge_gaps = Vec::new();
        let mut strength_areas = Vec::new();
        
        // Analyze concept coverage
        let mentioned_concepts: std::collections::HashSet<_> = history.interactions.iter()
            .flat_map(|interaction| {
                knowledge_base.core_concepts.iter()
                    .chain(knowledge_base.advanced_concepts.iter())
                    .filter(|&concept| interaction.query.to_lowercase().contains(&concept.to_lowercase()))
            })
            .collect();
        
        // Identify gaps in core concepts
        for concept in &knowledge_base.core_concepts {
            if !mentioned_concepts.contains(concept) && expertise_level < 0.8 {
                knowledge_gaps.push(format!("Limited exposure to core concept: {}", concept));
            }
        }
        
        // Identify strength areas
        for concept in &mentioned_concepts {
            if knowledge_base.advanced_concepts.contains(concept) {
                strength_areas.push(format!("Strong understanding of: {}", concept));
            }
        }
        
        // Add pattern-based gaps and strengths
        if expertise_level < 0.3 {
            knowledge_gaps.push("Basic conceptual understanding needed".to_string());
        } else if expertise_level > 0.7 {
            strength_areas.push("Advanced conceptual thinking demonstrated".to_string());
        }
        
        Ok((knowledge_gaps, strength_areas))
    }

    fn get_or_create_knowledge_base(&mut self, domain: &str) -> &DomainKnowledgeBase {
        if !self.expert_knowledge_base.contains_key(domain) {
            let knowledge_base = self.create_knowledge_base_for_domain(domain);
            self.expert_knowledge_base.insert(domain.to_string(), knowledge_base);
        }
        
        self.expert_knowledge_base.get(domain).unwrap()
    }

    fn create_knowledge_base_for_domain(&self, domain: &str) -> DomainKnowledgeBase {
        match domain {
            "physics" => DomainKnowledgeBase {
                domain_name: "physics".to_string(),
                core_concepts: vec![
                    "energy".to_string(), "force".to_string(), "momentum".to_string(),
                    "mass".to_string(), "velocity".to_string(), "acceleration".to_string(),
                ],
                advanced_concepts: vec![
                    "quantum mechanics".to_string(), "relativity".to_string(), "field theory".to_string(),
                    "thermodynamics".to_string(), "electromagnetism".to_string(),
                ],
                expert_vocabulary: vec![
                    "hamiltonian".to_string(), "lagrangian".to_string(), "eigenvalue".to_string(),
                    "wavefunction".to_string(), "spacetime".to_string(), "entropy".to_string(),
                ],
                common_misconceptions: vec![
                    "heavier objects fall faster".to_string(),
                    "centrifugal force is real".to_string(),
                    "heat and temperature are the same".to_string(),
                ],
                expertise_indicators: vec![
                    "derives equations".to_string(), "discusses symmetries".to_string(),
                    "mentions conservation laws".to_string(),
                ],
            },
            "computer_science" => DomainKnowledgeBase {
                domain_name: "computer_science".to_string(),
                core_concepts: vec![
                    "algorithm".to_string(), "data structure".to_string(), "complexity".to_string(),
                    "programming".to_string(), "software".to_string(), "hardware".to_string(),
                ],
                advanced_concepts: vec![
                    "machine learning".to_string(), "distributed systems".to_string(),
                    "cryptography".to_string(), "compiler design".to_string(),
                ],
                expert_vocabulary: vec![
                    "big-o notation".to_string(), "polymorphism".to_string(), "concurrency".to_string(),
                    "abstraction".to_string(), "encapsulation".to_string(),
                ],
                common_misconceptions: vec![
                    "more code means better".to_string(),
                    "premature optimization".to_string(),
                    "ai will replace programmers".to_string(),
                ],
                expertise_indicators: vec![
                    "discusses trade-offs".to_string(), "mentions design patterns".to_string(),
                    "considers scalability".to_string(),
                ],
            },
            _ => DomainKnowledgeBase {
                domain_name: domain.to_string(),
                core_concepts: vec!["basic".to_string(), "fundamental".to_string()],
                advanced_concepts: vec!["advanced".to_string(), "complex".to_string()],
                expert_vocabulary: vec!["technical".to_string(), "specialized".to_string()],
                common_misconceptions: vec!["common error".to_string()],
                expertise_indicators: vec!["shows expertise".to_string()],
            },
        }
    }

    fn initialize_knowledge_bases(&mut self) {
        // Initialize with common domains
        let domains = ["physics", "computer_science", "mathematics", "chemistry", "biology"];
        for domain in domains {
            let knowledge_base = self.create_knowledge_base_for_domain(domain);
            self.expert_knowledge_base.insert(domain.to_string(), knowledge_base);
        }
    }
}

impl ExpertisePatterns {
    fn new() -> Self {
        Self {
            novice_patterns: vec![
                "what is".to_string(),
                "how do i".to_string(),
                "can you explain".to_string(),
                "i don't understand".to_string(),
            ],
            intermediate_patterns: vec![
                "how does".to_string(),
                "what are the differences".to_string(),
                "can you compare".to_string(),
                "what happens when".to_string(),
            ],
            advanced_patterns: vec![
                "what are the implications".to_string(),
                "how might".to_string(),
                "what if we consider".to_string(),
                "from a theoretical perspective".to_string(),
            ],
            expert_patterns: vec![
                "given the constraints".to_string(),
                "considering the trade-offs".to_string(),
                "in the context of".to_string(),
                "what are the fundamental".to_string(),
            ],
        }
    }
}

/// Quality orchestrator for assessment validation (Side 3 of the four-sided triangle)
pub struct QualityOrchestrator {
    quality_thresholds: QualityThresholds,
    validation_history: Vec<ValidationRecord>,
}

#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub minimum_consistency: f64,
    pub minimum_reliability: f64,
    pub minimum_validity: f64,
    pub minimum_completeness: f64,
}

#[derive(Debug, Clone)]
pub struct ValidationRecord {
    pub timestamp: DateTime<Utc>,
    pub assessment_quality: f64,
    pub validation_passed: bool,
    pub issues_found: Vec<String>,
}

impl QualityOrchestrator {
    pub fn new() -> Self {
        Self {
            quality_thresholds: QualityThresholds {
                minimum_consistency: 0.7,
                minimum_reliability: 0.75,
                minimum_validity: 0.8,
                minimum_completeness: 0.6,
            },
            validation_history: Vec::new(),
        }
    }

    pub async fn validate_assessment_quality(
        &mut self,
        model_assessments: &[ModelAssessment],
        expert_knowledge: &DomainExpertise,
    ) -> Result<QualityMetrics> {
        debug!("Validating assessment quality across {} models", model_assessments.len());

        let consistency_score = self.calculate_consistency_score(model_assessments)?;
        let reliability_score = self.calculate_reliability_score(model_assessments)?;
        let validity_score = self.calculate_validity_score(model_assessments, expert_knowledge)?;
        let completeness_score = self.calculate_completeness_score(model_assessments, expert_knowledge)?;
        
        let overall_quality = (consistency_score + reliability_score + validity_score + completeness_score) / 4.0;
        
        let quality_metrics = QualityMetrics {
            overall_quality,
            consistency_score,
            reliability_score,
            validity_score,
            assessment_completeness: completeness_score,
        };

        // Record validation
        let validation_passed = self.meets_quality_thresholds(&quality_metrics);
        self.validation_history.push(ValidationRecord {
            timestamp: Utc::now(),
            assessment_quality: overall_quality,
            validation_passed,
            issues_found: self.identify_quality_issues(&quality_metrics),
        });

        Ok(quality_metrics)
    }

    fn calculate_consistency_score(&self, assessments: &[ModelAssessment]) -> Result<f64> {
        if assessments.len() < 2 {
            return Ok(0.5); // Can't measure consistency with less than 2 assessments
        }

        let scores: Vec<f64> = assessments.iter().map(|a| a.assessment_score).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        
        let variance = scores.iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>() / scores.len() as f64;
        
        let std_dev = variance.sqrt();
        
        // Consistency is higher when standard deviation is lower
        let consistency = 1.0 - (std_dev / 1.0).min(1.0);
        Ok(consistency)
    }

    fn calculate_reliability_score(&self, assessments: &[ModelAssessment]) -> Result<f64> {
        let avg_confidence = assessments.iter()
            .map(|a| a.confidence)
            .sum::<f64>() / assessments.len().max(1) as f64;
        
        let avg_processing_time = assessments.iter()
            .map(|a| a.processing_time_ms as f64)
            .sum::<f64>() / assessments.len().max(1) as f64;
        
        // Reliability considers both confidence and reasonable processing time
        let time_factor = if avg_processing_time > 5000.0 { 0.8 } else { 1.0 }; // Penalty for very slow processing
        
        Ok(avg_confidence * time_factor)
    }

    fn calculate_validity_score(&self, assessments: &[ModelAssessment], expert_knowledge: &DomainExpertise) -> Result<f64> {
        // Validity is how well the assessments align with expert knowledge
        let avg_assessment = assessments.iter()
            .map(|a| a.assessment_score)
            .sum::<f64>() / assessments.len().max(1) as f64;
        
        let expert_level = expert_knowledge.expertise_level;
        
        // Validity is higher when assessment aligns with expert indicators
        let alignment = 1.0 - (avg_assessment - expert_level).abs();
        
        // Bonus for having evidence
        let evidence_bonus = if assessments.iter().any(|a| !a.evidence_points.is_empty()) { 0.1 } else { 0.0 };
        
        Ok((alignment + evidence_bonus).min(1.0))
    }

    fn calculate_completeness_score(&self, assessments: &[ModelAssessment], expert_knowledge: &DomainExpertise) -> Result<f64> {
        let mut completeness_factors = Vec::new();
        
        // Check if we have multiple model types
        let model_diversity = assessments.len() as f64 / 5.0; // Assuming max 5 different model types
        completeness_factors.push(model_diversity.min(1.0));
        
        // Check if we have evidence from assessments
        let evidence_coverage = assessments.iter()
            .filter(|a| !a.evidence_points.is_empty())
            .count() as f64 / assessments.len().max(1) as f64;
        completeness_factors.push(evidence_coverage);
        
        // Check if expert knowledge has sufficient indicators
        let expert_indicator_coverage = if expert_knowledge.expert_patterns.is_empty() { 0.5 } else { 1.0 };
        completeness_factors.push(expert_indicator_coverage);
        
        let completeness = completeness_factors.iter().sum::<f64>() / completeness_factors.len() as f64;
        Ok(completeness)
    }

    fn meets_quality_thresholds(&self, metrics: &QualityMetrics) -> bool {
        metrics.consistency_score >= self.quality_thresholds.minimum_consistency &&
        metrics.reliability_score >= self.quality_thresholds.minimum_reliability &&
        metrics.validity_score >= self.quality_thresholds.minimum_validity &&
        metrics.assessment_completeness >= self.quality_thresholds.minimum_completeness
    }

    fn identify_quality_issues(&self, metrics: &QualityMetrics) -> Vec<String> {
        let mut issues = Vec::new();
        
        if metrics.consistency_score < self.quality_thresholds.minimum_consistency {
            issues.push(format!("Low consistency: {:.2} < {:.2}", 
                              metrics.consistency_score, self.quality_thresholds.minimum_consistency));
        }
        
        if metrics.reliability_score < self.quality_thresholds.minimum_reliability {
            issues.push(format!("Low reliability: {:.2} < {:.2}", 
                              metrics.reliability_score, self.quality_thresholds.minimum_reliability));
        }
        
        if metrics.validity_score < self.quality_thresholds.minimum_validity {
            issues.push(format!("Low validity: {:.2} < {:.2}", 
                              metrics.validity_score, self.quality_thresholds.minimum_validity));
        }
        
        if metrics.assessment_completeness < self.quality_thresholds.minimum_completeness {
            issues.push(format!("Incomplete assessment: {:.2} < {:.2}", 
                              metrics.assessment_completeness, self.quality_thresholds.minimum_completeness));
        }
        
        issues
    }
}
