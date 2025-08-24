use crate::*;
use anyhow::Result;
use tracing::{debug, warn};

/// Security validator implementing ecosystem uniqueness security model
pub struct SecurityValidator {
    uniqueness_threshold: f64,
    security_cache: HashMap<String, f64>,
}

impl SecurityValidator {
    pub fn new() -> Self {
        Self {
            uniqueness_threshold: 0.95,
            security_cache: HashMap::new(),
        }
    }

    /// Calculate ecosystem uniqueness for security validation
    /// Security emerges from the impossibility of replicating the complete ecosystem
    pub fn calculate_ecosystem_uniqueness(
        &self,
        person_signature: &PersonSignature,
        machine_signature: &EcosystemSignature,
    ) -> Result<f64> {
        debug!("Calculating ecosystem uniqueness for security validation");

        // Person uniqueness component
        let person_uniqueness = self.calculate_person_uniqueness(person_signature)?;
        
        // Machine uniqueness component  
        let machine_uniqueness = machine_signature.calculate_uniqueness();
        
        // Interaction uniqueness (combination of person + machine)
        let interaction_uniqueness = self.calculate_interaction_uniqueness(
            person_signature, 
            machine_signature
        )?;
        
        // Temporal uniqueness (time-based patterns)
        let temporal_uniqueness = self.calculate_temporal_uniqueness(person_signature)?;

        // Combined ecosystem uniqueness (geometric mean for security)
        let ecosystem_uniqueness = (
            person_uniqueness * 
            machine_uniqueness * 
            interaction_uniqueness * 
            temporal_uniqueness
        ).powf(0.25); // 4th root for geometric mean

        debug!("Ecosystem uniqueness components: person={:.3}, machine={:.3}, interaction={:.3}, temporal={:.3}, combined={:.3}",
               person_uniqueness, machine_uniqueness, interaction_uniqueness, temporal_uniqueness, ecosystem_uniqueness);

        Ok(ecosystem_uniqueness)
    }

    fn calculate_person_uniqueness(&self, person_signature: &PersonSignature) -> Result<f64> {
        let behavioral_diversity = person_signature.behavioral_patterns.len() as f64 / 20.0; // Normalize to 20 patterns
        let pattern_complexity = person_signature.behavioral_patterns.values()
            .map(|&v| v.abs())
            .sum::<f64>() / person_signature.behavioral_patterns.len().max(1) as f64;
        let consistency_factor = person_signature.preference_consistency;
        let temporal_complexity = person_signature.temporal_patterns.len() as f64 / 24.0; // 24 hours

        let uniqueness = (
            behavioral_diversity * 0.3 +
            pattern_complexity * 0.3 +
            consistency_factor * 0.2 +
            temporal_complexity * 0.2
        ).min(1.0);

        Ok(uniqueness)
    }

    fn calculate_interaction_uniqueness(
        &self,
        person_signature: &PersonSignature,
        machine_signature: &EcosystemSignature,
    ) -> Result<f64> {
        // Interaction uniqueness emerges from the specific combination of person + machine
        let person_machine_coupling = self.calculate_coupling_strength(person_signature, machine_signature)?;
        let interaction_complexity = self.calculate_interaction_complexity(person_signature)?;
        let ecosystem_coherence = self.calculate_ecosystem_coherence(person_signature, machine_signature)?;

        let uniqueness = (person_machine_coupling * interaction_complexity * ecosystem_coherence).powf(1.0/3.0);
        Ok(uniqueness.min(1.0))
    }

    fn calculate_coupling_strength(
        &self,
        person_signature: &PersonSignature,
        machine_signature: &EcosystemSignature,
    ) -> Result<f64> {
        // How well the person's patterns match the machine's capabilities
        let style_match = match person_signature.interaction_style.as_str() {
            style if style.contains("Technical") => machine_signature.machine_fingerprint.performance_profile.cpu_characteristics.len() as f64 / 100.0,
            style if style.contains("Creative") => machine_signature.environment_signature.context_stability,
            style if style.contains("Direct") => machine_signature.interaction_patterns.preference_stability,
            _ => 0.7, // Default coupling
        };

        let temporal_sync = if person_signature.temporal_patterns.len() > 0 && machine_signature.temporal_signature.session_patterns.len() > 0 {
            let person_avg = person_signature.temporal_patterns.iter().sum::<f64>() / person_signature.temporal_patterns.len() as f64;
            let machine_avg = machine_signature.temporal_signature.session_patterns.iter().sum::<f64>() / machine_signature.temporal_signature.session_patterns.len() as f64;
            1.0 - (person_avg - machine_avg).abs().min(1.0)
        } else {
            0.5
        };

        Ok((style_match + temporal_sync) / 2.0)
    }

    fn calculate_interaction_complexity(&self, person_signature: &PersonSignature) -> Result<f64> {
        let pattern_variance = if person_signature.behavioral_patterns.len() > 1 {
            let mean = person_signature.behavioral_patterns.values().sum::<f64>() / person_signature.behavioral_patterns.len() as f64;
            let variance = person_signature.behavioral_patterns.values()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>() / person_signature.behavioral_patterns.len() as f64;
            variance.sqrt().min(1.0)
        } else {
            0.5
        };

        let temporal_variance = if person_signature.temporal_patterns.len() > 1 {
            let mean = person_signature.temporal_patterns.iter().sum::<f64>() / person_signature.temporal_patterns.len() as f64;
            let variance = person_signature.temporal_patterns.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>() / person_signature.temporal_patterns.len() as f64;
            variance.sqrt().min(1.0)
        } else {
            0.5
        };

        Ok((pattern_variance + temporal_variance) / 2.0)
    }

    fn calculate_ecosystem_coherence(
        &self,
        person_signature: &PersonSignature,
        machine_signature: &EcosystemSignature,
    ) -> Result<f64> {
        // How coherently the person-machine ecosystem functions as a unit
        let consistency_alignment = (person_signature.preference_consistency + machine_signature.interaction_patterns.preference_stability) / 2.0;
        let environmental_stability = machine_signature.environment_signature.context_stability;
        let system_maturity = (machine_signature.temporal_signature.session_patterns.len() as f64 / 100.0).min(1.0);

        Ok((consistency_alignment + environmental_stability + system_maturity) / 3.0)
    }

    fn calculate_temporal_uniqueness(&self, person_signature: &PersonSignature) -> Result<f64> {
        if person_signature.temporal_patterns.is_empty() {
            return Ok(0.5); // Default for new users
        }

        // Analyze temporal pattern uniqueness
        let pattern_entropy = self.calculate_pattern_entropy(&person_signature.temporal_patterns)?;
        let pattern_stability = person_signature.preference_consistency;
        let temporal_complexity = (person_signature.temporal_patterns.len() as f64 / 24.0).min(1.0);

        Ok((pattern_entropy + pattern_stability + temporal_complexity) / 3.0)
    }

    fn calculate_pattern_entropy(&self, patterns: &[f64]) -> Result<f64> {
        if patterns.is_empty() {
            return Ok(0.0);
        }

        // Simple entropy calculation for temporal patterns
        let sum: f64 = patterns.iter().sum();
        if sum == 0.0 {
            return Ok(0.0);
        }

        let entropy = patterns.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| {
                let prob = p / sum;
                -prob * prob.log2()
            })
            .sum::<f64>();

        // Normalize entropy to [0, 1]
        let max_entropy = (patterns.len() as f64).log2();
        Ok(if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 })
    }

    /// Validate that ecosystem uniqueness meets security threshold
    pub fn validate_security_threshold(
        &self,
        person_signature: &PersonSignature,
        machine_signature: &EcosystemSignature,
    ) -> Result<bool> {
        let uniqueness = self.calculate_ecosystem_uniqueness(person_signature, machine_signature)?;
        
        if uniqueness >= self.uniqueness_threshold {
            debug!("Security validation passed: {:.3} >= {:.3}", uniqueness, self.uniqueness_threshold);
            Ok(true)
        } else {
            warn!("Security validation failed: {:.3} < {:.3}", uniqueness, self.uniqueness_threshold);
            Ok(false)
        }
    }

    /// Get security metrics for monitoring
    pub fn get_security_metrics(
        &self,
        person_signature: &PersonSignature,
        machine_signature: &EcosystemSignature,
    ) -> Result<SecurityMetrics> {
        let ecosystem_uniqueness = self.calculate_ecosystem_uniqueness(person_signature, machine_signature)?;
        let person_uniqueness = self.calculate_person_uniqueness(person_signature)?;
        let machine_uniqueness = machine_signature.calculate_uniqueness();
        let interaction_uniqueness = self.calculate_interaction_uniqueness(person_signature, machine_signature)?;
        let temporal_uniqueness = self.calculate_temporal_uniqueness(person_signature)?;

        Ok(SecurityMetrics {
            ecosystem_uniqueness,
            person_uniqueness,
            machine_uniqueness,
            interaction_uniqueness,
            temporal_uniqueness,
            security_threshold: self.uniqueness_threshold,
            security_status: ecosystem_uniqueness >= self.uniqueness_threshold,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub ecosystem_uniqueness: f64,
    pub person_uniqueness: f64,
    pub machine_uniqueness: f64,
    pub interaction_uniqueness: f64,
    pub temporal_uniqueness: f64,
    pub security_threshold: f64,
    pub security_status: bool,
}

/// Attack vector analysis for ecosystem security
pub struct AttackVectorAnalyzer;

impl AttackVectorAnalyzer {
    /// Analyze the difficulty of ecosystem impersonation
    pub fn analyze_impersonation_difficulty(
        person_signature: &PersonSignature,
        machine_signature: &EcosystemSignature,
    ) -> ImpersonationDifficulty {
        let behavioral_complexity = person_signature.behavioral_patterns.len();
        let temporal_complexity = person_signature.temporal_patterns.len();
        let machine_complexity = machine_signature.machine_fingerprint.performance_profile.network_latency_profile.len();
        let interaction_complexity = machine_signature.interaction_patterns.typing_patterns.keystroke_dynamics.len();

        let total_complexity = behavioral_complexity + temporal_complexity + machine_complexity + interaction_complexity;

        let difficulty_score = match total_complexity {
            0..=10 => 0.3,
            11..=25 => 0.6,
            26..=50 => 0.8,
            51..=100 => 0.9,
            _ => 0.95,
        };

        ImpersonationDifficulty {
            difficulty_score,
            required_behavioral_replication: behavioral_complexity,
            required_machine_replication: machine_complexity,
            required_temporal_consistency: temporal_complexity,
            required_interaction_mimicking: interaction_complexity,
            estimated_attack_cost: Self::estimate_attack_cost(difficulty_score),
        }
    }

    fn estimate_attack_cost(difficulty_score: f64) -> AttackCost {
        match difficulty_score {
            score if score < 0.5 => AttackCost::Low,
            score if score < 0.7 => AttackCost::Medium,
            score if score < 0.9 => AttackCost::High,
            _ => AttackCost::Prohibitive,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImpersonationDifficulty {
    pub difficulty_score: f64,
    pub required_behavioral_replication: usize,
    pub required_machine_replication: usize,
    pub required_temporal_consistency: usize,
    pub required_interaction_mimicking: usize,
    pub estimated_attack_cost: AttackCost,
}

#[derive(Debug, Clone)]
pub enum AttackCost {
    Low,        // Basic script kiddie level
    Medium,     // Requires some expertise
    High,       // Requires significant resources
    Prohibitive, // Practically impossible
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_validator_creation() {
        let validator = SecurityValidator::new();
        assert_eq!(validator.uniqueness_threshold, 0.95);
    }

    #[test]
    fn test_ecosystem_uniqueness_calculation() {
        let validator = SecurityValidator::new();
        let person_signature = PersonSignature::default();
        let machine_signature = EcosystemSignature::generate_unique();

        let uniqueness = validator.calculate_ecosystem_uniqueness(&person_signature, &machine_signature);
        assert!(uniqueness.is_ok());
        
        let uniqueness_value = uniqueness.unwrap();
        assert!(uniqueness_value >= 0.0 && uniqueness_value <= 1.0);
    }

    #[test]
    fn test_security_threshold_validation() {
        let validator = SecurityValidator::new();
        
        // Create a signature with high uniqueness
        let mut person_signature = PersonSignature::default();
        person_signature.behavioral_patterns.insert("pattern1".to_string(), 0.8);
        person_signature.behavioral_patterns.insert("pattern2".to_string(), 0.6);
        person_signature.behavioral_patterns.insert("pattern3".to_string(), 0.9);
        person_signature.preference_consistency = 0.9;
        person_signature.temporal_patterns = vec![0.8, 0.7, 0.9, 0.6, 0.8];

        let machine_signature = EcosystemSignature::generate_unique();

        let validation_result = validator.validate_security_threshold(&person_signature, &machine_signature);
        assert!(validation_result.is_ok());
    }

    #[test]
    fn test_attack_vector_analysis() {
        let person_signature = PersonSignature::default();
        let machine_signature = EcosystemSignature::generate_unique();

        let difficulty = AttackVectorAnalyzer::analyze_impersonation_difficulty(&person_signature, &machine_signature);
        
        assert!(difficulty.difficulty_score >= 0.0 && difficulty.difficulty_score <= 1.0);
        assert!(matches!(difficulty.estimated_attack_cost, AttackCost::Low | AttackCost::Medium | AttackCost::High | AttackCost::Prohibitive));
    }

    #[test]
    fn test_security_metrics() {
        let validator = SecurityValidator::new();
        let person_signature = PersonSignature::default();
        let machine_signature = EcosystemSignature::generate_unique();

        let metrics = validator.get_security_metrics(&person_signature, &machine_signature);
        assert!(metrics.is_ok());
        
        let metrics = metrics.unwrap();
        assert!(metrics.ecosystem_uniqueness >= 0.0 && metrics.ecosystem_uniqueness <= 1.0);
        assert!(metrics.person_uniqueness >= 0.0 && metrics.person_uniqueness <= 1.0);
        assert!(metrics.machine_uniqueness >= 0.0 && metrics.machine_uniqueness <= 1.0);
        assert_eq!(metrics.security_threshold, 0.95);
    }
}
