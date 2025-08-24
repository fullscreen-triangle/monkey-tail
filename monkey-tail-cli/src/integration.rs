use crate::*;
use anyhow::Result;
use tracing::{info, debug};
use uuid::Uuid;

/// Complete Monkey-Tail integration demonstrating all components working together
pub struct MonkeyTailIntegration {
    identity_processor: EphemeralIdentityProcessor,
    kambuzuma_processor: PersonalizedKambuzumaProcessor,
    competency_assessor: FourSidedTriangleAssessor,
    security_validator: SecurityValidator,
    user_sessions: HashMap<Uuid, UserSession>,
}

#[derive(Debug, Clone)]
pub struct UserSession {
    pub user_id: Uuid,
    pub session_start: chrono::DateTime<chrono::Utc>,
    pub interaction_count: u32,
    pub current_domain: String,
    pub expertise_progression: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct CompleteProcessingResult {
    pub semantic_identity: SemanticIdentity,
    pub processing_result: PersonalizedProcessingResult,
    pub competency_assessment: CompetencyAssessment,
    pub security_metrics: SecurityMetrics,
    pub session_info: UserSession,
}

impl MonkeyTailIntegration {
    pub async fn new() -> Result<Self> {
        info!("🚀 Initializing complete Monkey-Tail integration");
        
        Ok(Self {
            identity_processor: EphemeralIdentityProcessor::new(),
            kambuzuma_processor: PersonalizedKambuzumaProcessor::new().await?,
            competency_assessor: FourSidedTriangleAssessor::new(),
            security_validator: SecurityValidator::new(),
            user_sessions: HashMap::new(),
        })
    }

    /// Process a complete query through all Monkey-Tail components
    pub async fn process_complete_query(
        &mut self,
        query: &str,
        domain: &str,
        user_expertise_level: f64,
    ) -> Result<CompleteProcessingResult> {
        let user_id = Uuid::new_v4(); // In practice, this would be from authentication
        
        info!("🔄 Processing complete query for user {} in domain '{}'", user_id, domain);
        
        // Step 1: Create or update user session
        let session = self.get_or_create_session(user_id, domain).await?;
        
        // Step 2: Extract ephemeral semantic identity
        let interaction_data = InteractionData {
            query_complexity: self.estimate_query_complexity(query),
            domain_context: domain.to_string(),
            user_expertise_level,
            communication_preferences: CommunicationPreferences::default(),
            urgency_level: UrgencyLevel::Medium,
            expected_response_type: ResponseType::DirectAnswer,
        };
        
        let semantic_identity = self.identity_processor
            .extract_current_identity(user_id, &interaction_data)
            .await?;
        
        debug!("✅ Extracted semantic identity with {:.1}% BMD effectiveness", 
               semantic_identity.calculate_bmd_effectiveness(domain) * 100.0);

        // Step 3: Assess competency using four-sided triangle
        let interaction_history = self.build_interaction_history(&session)?;
        let competency_assessment = self.competency_assessor
            .assess_competency(query, domain, &interaction_history)
            .await?;
        
        debug!("✅ Competency assessment: {:.1}% understanding", 
               competency_assessment.understanding_level * 100.0);

        // Step 4: Process through personalized Kambuzuma
        let personalized_input = PersonalizedStageInput {
            query: query.to_string(),
            context: Some(format!("User session in {}", domain)),
            user_semantic_identity: semantic_identity.clone(),
            interaction_data: interaction_data.clone(),
            environmental_context: EnvironmentalContext::default(),
            timestamp: chrono::Utc::now(),
        };
        
        let processing_result = self.kambuzuma_processor
            .process_query_with_semantic_identity(user_id, personalized_input)
            .await?;
        
        debug!("✅ Kambuzuma processing: {:.1}% BMD effectiveness, {}ms", 
               processing_result.bmd_effectiveness * 100.0,
               processing_result.processing_time_ms);

        // Step 5: Validate ecosystem security
        let person_signature = self.calculate_person_signature(user_id, &semantic_identity).await?;
        let machine_signature = self.identity_processor.get_ecosystem_signature();
        
        let security_metrics = self.security_validator
            .get_security_metrics(&person_signature, machine_signature)?;
        
        debug!("✅ Security validation: {:.1}% ecosystem uniqueness", 
               security_metrics.ecosystem_uniqueness * 100.0);

        // Step 6: Update identity from interaction results
        self.identity_processor
            .update_identity_from_interaction(
                user_id,
                query,
                &processing_result,
                &processing_result.learning_insights,
            )
            .await?;

        // Step 7: Update session tracking
        self.update_session_tracking(user_id, domain, &competency_assessment).await?;

        let updated_session = self.user_sessions.get(&user_id).unwrap().clone();

        info!("🎉 Complete processing finished successfully");

        Ok(CompleteProcessingResult {
            semantic_identity,
            processing_result,
            competency_assessment,
            security_metrics,
            session_info: updated_session,
        })
    }

    async fn get_or_create_session(&mut self, user_id: Uuid, domain: &str) -> Result<UserSession> {
        if let Some(session) = self.user_sessions.get_mut(&user_id) {
            session.interaction_count += 1;
            session.current_domain = domain.to_string();
            Ok(session.clone())
        } else {
            let new_session = UserSession {
                user_id,
                session_start: chrono::Utc::now(),
                interaction_count: 1,
                current_domain: domain.to_string(),
                expertise_progression: HashMap::new(),
            };
            
            self.user_sessions.insert(user_id, new_session.clone());
            Ok(new_session)
        }
    }

    fn estimate_query_complexity(&self, query: &str) -> f64 {
        let word_count = query.split_whitespace().count() as f64;
        let char_count = query.len() as f64;
        let question_words = ["what", "how", "why", "when", "where", "which"];
        let has_question_word = question_words.iter().any(|&word| query.to_lowercase().contains(word));
        
        let technical_words = ["algorithm", "implementation", "optimization", "analysis", "synthesis"];
        let technical_count = technical_words.iter()
            .filter(|&word| query.to_lowercase().contains(word))
            .count() as f64;
        
        // Complexity estimation
        let length_factor = (word_count / 20.0).min(1.0);
        let complexity_factor = if has_question_word { 0.7 } else { 0.5 };
        let technical_factor = (technical_count / technical_words.len() as f64).min(0.3);
        let char_factor = (char_count / 200.0).min(0.2);
        
        (length_factor + complexity_factor + technical_factor + char_factor).min(1.0)
    }

    fn build_interaction_history(&self, session: &UserSession) -> Result<InteractionHistory> {
        // In a real implementation, this would build from actual interaction history
        // For demo purposes, we create a mock history based on session info
        
        let mut interactions = Vec::new();
        
        // Create mock interactions based on session data
        for i in 0..session.interaction_count.min(5) {
            interactions.push(HistoricalInteraction {
                timestamp: session.session_start + chrono::Duration::minutes(i as i64 * 5),
                query: format!("Mock query {} in {}", i + 1, session.current_domain),
                domain: session.current_domain.clone(),
                complexity: 0.5 + (i as f64 * 0.1),
                success_indicators: vec!["engagement".to_string(), "understanding".to_string()],
            });
        }
        
        Ok(InteractionHistory {
            interactions,
            total_interactions: session.interaction_count as usize,
            domains_covered: vec![session.current_domain.clone()],
            average_complexity: 0.6,
        })
    }

    async fn calculate_person_signature(&self, user_id: Uuid, semantic_identity: &SemanticIdentity) -> Result<PersonSignature> {
        let session = self.user_sessions.get(&user_id);
        
        let mut behavioral_patterns = HashMap::new();
        
        if let Some(session) = session {
            behavioral_patterns.insert("interaction_frequency".to_string(), session.interaction_count as f64 / 10.0);
            behavioral_patterns.insert("domain_consistency".to_string(), 0.8); // Mock consistency
            behavioral_patterns.insert("session_length".to_string(), 
                chrono::Utc::now().signed_duration_since(session.session_start).num_minutes() as f64 / 60.0);
        }
        
        // Add patterns from semantic identity
        for (domain, understanding) in &semantic_identity.understanding_vector.domains {
            behavioral_patterns.insert(format!("understanding_{}", domain), *understanding);
        }
        
        let interaction_style = format!("{:?}", semantic_identity.communication_patterns.communication_style);
        let preference_consistency = semantic_identity.understanding_vector.domains.len() as f64 / 10.0;
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

    async fn update_session_tracking(&mut self, user_id: Uuid, domain: &str, assessment: &CompetencyAssessment) -> Result<()> {
        if let Some(session) = self.user_sessions.get_mut(&user_id) {
            // Track expertise progression
            session.expertise_progression
                .entry(domain.to_string())
                .or_insert_with(Vec::new)
                .push(assessment.understanding_level);
            
            // Keep only last 10 assessments per domain
            if let Some(progression) = session.expertise_progression.get_mut(domain) {
                if progression.len() > 10 {
                    progression.remove(0);
                }
            }
        }
        
        Ok(())
    }

    /// Get comprehensive system status
    pub async fn get_system_status(&self) -> Result<SystemStatus> {
        let identity_stats = self.identity_processor.get_session_stats().await?;
        let kambuzuma_stats = self.kambuzuma_processor.get_processing_stats().await;
        let competency_stats = self.competency_assessor.get_assessment_stats().await?;
        
        Ok(SystemStatus {
            active_sessions: self.user_sessions.len(),
            total_interactions: identity_stats.total_interactions,
            average_bmd_effectiveness: kambuzuma_stats.average_bmd_effectiveness,
            average_processing_time_ms: kambuzuma_stats.average_processing_time_ms,
            security_validations_passed: 0, // Would track in real implementation
            domains_active: self.user_sessions.values()
                .map(|s| s.current_domain.clone())
                .collect::<std::collections::HashSet<_>>()
                .len(),
        })
    }

    /// Clear all ephemeral data (privacy protection)
    pub async fn clear_all_ephemeral_data(&mut self) -> Result<()> {
        info!("🧹 Clearing all ephemeral data for privacy protection");
        
        // Clear identity processor data for all users
        for user_id in self.user_sessions.keys() {
            self.identity_processor.clear_user_data(*user_id).await?;
        }
        
        // Clear session tracking
        self.user_sessions.clear();
        
        info!("✅ All ephemeral data cleared successfully");
        Ok(())
    }

    /// Demonstrate the "One Machine, One User, One Application" principle
    pub async fn demonstrate_omua_principle(&mut self) -> Result<()> {
        info!("🔒 Demonstrating One Machine, One User, One Application principle");
        
        // Generate unique ecosystem signature
        let machine_signature = self.identity_processor.get_ecosystem_signature();
        let machine_uniqueness = machine_signature.calculate_uniqueness();
        
        println!("Machine Ecosystem Signature:");
        println!("  Hardware ID: {}", &machine_signature.machine_fingerprint.hardware_id[..16]);
        println!("  OS Signature: {}", machine_signature.machine_fingerprint.os_signature);
        println!("  Network Signature: {}", &machine_signature.machine_fingerprint.network_signature[..16]);
        println!("  Uniqueness Score: {:.1}%", machine_uniqueness * 100.0);
        println!();
        
        // Demonstrate that this creates a unique environment
        println!("🎯 OMUA Principle Benefits:");
        println!("✅ Dedicated processing power for single user");
        println!("✅ No data mixing between users");
        println!("✅ Personalized BMD effectiveness scaling");
        println!("✅ Ecosystem security through uniqueness");
        println!("✅ Zero persistent storage of personal data");
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub active_sessions: usize,
    pub total_interactions: usize,
    pub average_bmd_effectiveness: f64,
    pub average_processing_time_ms: u64,
    pub security_validations_passed: usize,
    pub domains_active: usize,
}

impl SystemStatus {
    pub fn print_status(&self) {
        println!("📊 Monkey-Tail System Status");
        println!("Active Sessions: {}", self.active_sessions);
        println!("Total Interactions: {}", self.total_interactions);
        println!("Average BMD Effectiveness: {:.1}%", self.average_bmd_effectiveness * 100.0);
        println!("Average Processing Time: {}ms", self.average_processing_time_ms);
        println!("Security Validations: {}", self.security_validations_passed);
        println!("Active Domains: {}", self.domains_active);
    }
}
