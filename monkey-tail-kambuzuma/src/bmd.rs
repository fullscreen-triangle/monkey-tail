use crate::*;
use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, info};

/// BMD (Biological Maxwell Demon) orchestrator for frame selection and processing
pub struct BmdOrchestrator {
    frame_database: FrameDatabase,
    selection_history: Vec<FrameSelection>,
    effectiveness_cache: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct FrameDatabase {
    frames: HashMap<String, CognitiveFrame>,
    domain_mappings: HashMap<String, Vec<String>>, // domain -> frame_ids
}

#[derive(Debug, Clone)]
pub struct CognitiveFrame {
    pub id: String,
    pub domain: String,
    pub complexity_level: f64,
    pub effectiveness_score: f64,
    pub usage_count: u64,
    pub success_rate: f64,
    pub frame_content: FrameContent,
}

#[derive(Debug, Clone)]
pub struct FrameContent {
    pub approach: String,
    pub key_concepts: Vec<String>,
    pub explanation_style: String,
    pub example_types: Vec<String>,
    pub prerequisite_knowledge: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FrameSelection {
    pub frame_id: String,
    pub selection_confidence: f64,
    pub user_expertise_match: f64,
    pub context_relevance: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct BmdProcessingResult {
    pub content: String,
    pub confidence: f64,
    pub frames_considered: u32,
    pub navigation_steps: u32,
    pub selected_frame: CognitiveFrame,
}

impl BmdOrchestrator {
    pub fn new() -> Self {
        let mut orchestrator = Self {
            frame_database: FrameDatabase::new(),
            selection_history: Vec::new(),
            effectiveness_cache: HashMap::new(),
        };
        
        // Initialize with some basic cognitive frames
        orchestrator.initialize_basic_frames();
        orchestrator
    }

    pub async fn process_with_bmd_effectiveness(
        &mut self,
        input: &PersonalizedStageInput,
        bmd_effectiveness: f64,
    ) -> Result<BmdProcessingResult> {
        debug!("BMD processing with effectiveness: {:.2}%", bmd_effectiveness * 100.0);

        // Select optimal cognitive frame based on user context
        let selected_frame = self.select_optimal_frame(input, bmd_effectiveness).await?;
        
        // Process query through selected frame
        let processed_content = self.process_through_frame(
            &input.query,
            &selected_frame,
            bmd_effectiveness,
        ).await?;

        // Update frame statistics
        self.update_frame_statistics(&selected_frame.id, true).await;

        // Record selection for learning
        self.selection_history.push(FrameSelection {
            frame_id: selected_frame.id.clone(),
            selection_confidence: bmd_effectiveness,
            user_expertise_match: input.interaction_data.user_expertise_level,
            context_relevance: self.calculate_context_relevance(input, &selected_frame),
            timestamp: chrono::Utc::now(),
        });

        Ok(BmdProcessingResult {
            content: processed_content,
            confidence: bmd_effectiveness * selected_frame.effectiveness_score,
            frames_considered: self.count_considered_frames(&input.interaction_data.domain_context),
            navigation_steps: self.estimate_navigation_steps(bmd_effectiveness),
            selected_frame,
        })
    }

    async fn select_optimal_frame(
        &mut self,
        input: &PersonalizedStageInput,
        bmd_effectiveness: f64,
    ) -> Result<CognitiveFrame> {
        let domain = &input.interaction_data.domain_context;
        let user_expertise = input.interaction_data.user_expertise_level;
        
        // Get candidate frames for this domain
        let candidate_frames = self.frame_database.get_frames_for_domain(domain);
        
        if candidate_frames.is_empty() {
            // Create a default frame for unknown domains
            return Ok(self.create_default_frame(domain, user_expertise));
        }

        // Score each frame based on user context and BMD effectiveness
        let mut best_frame = candidate_frames[0].clone();
        let mut best_score = 0.0;

        for frame in candidate_frames {
            let score = self.calculate_frame_score(frame, input, bmd_effectiveness);
            if score > best_score {
                best_score = score;
                best_frame = frame.clone();
            }
        }

        debug!("Selected frame '{}' with score {:.2}", best_frame.id, best_score);
        Ok(best_frame)
    }

    fn calculate_frame_score(
        &self,
        frame: &CognitiveFrame,
        input: &PersonalizedStageInput,
        bmd_effectiveness: f64,
    ) -> f64 {
        let expertise_match = 1.0 - (frame.complexity_level - input.interaction_data.user_expertise_level).abs();
        let effectiveness_bonus = bmd_effectiveness * 0.3;
        let usage_bonus = (frame.success_rate * frame.usage_count as f64 / 100.0).min(0.2);
        let context_relevance = self.calculate_context_relevance(input, frame);
        
        expertise_match * 0.4 + effectiveness_bonus + usage_bonus + context_relevance * 0.3
    }

    fn calculate_context_relevance(&self, input: &PersonalizedStageInput, frame: &CognitiveFrame) -> f64 {
        // Simplified context relevance calculation
        let query_complexity_match = 1.0 - (frame.complexity_level - input.interaction_data.query_complexity).abs();
        let style_match = match input.interaction_data.communication_preferences.style {
            CommunicationStyle::Technical => if frame.frame_content.approach.contains("technical") { 1.0 } else { 0.5 },
            CommunicationStyle::Creative => if frame.frame_content.approach.contains("creative") { 1.0 } else { 0.5 },
            CommunicationStyle::Direct => if frame.frame_content.approach.contains("direct") { 1.0 } else { 0.7 },
            _ => 0.8,
        };
        
        (query_complexity_match + style_match) / 2.0
    }

    async fn process_through_frame(
        &self,
        query: &str,
        frame: &CognitiveFrame,
        bmd_effectiveness: f64,
    ) -> Result<String> {
        // Mock processing - in real implementation this would interface with actual Kambuzuma
        let base_response = format!(
            "Processing query '{}' through {} frame with {:.1}% BMD effectiveness",
            query, frame.frame_content.approach, bmd_effectiveness * 100.0
        );

        // Enhance response based on frame content
        let enhanced_response = format!(
            "{}\n\nApproach: {}\nKey concepts: {}\nStyle: {}",
            base_response,
            frame.frame_content.approach,
            frame.frame_content.key_concepts.join(", "),
            frame.frame_content.explanation_style
        );

        Ok(enhanced_response)
    }

    async fn update_frame_statistics(&mut self, frame_id: &str, success: bool) {
        if let Some(frame) = self.frame_database.frames.get_mut(frame_id) {
            frame.usage_count += 1;
            if success {
                frame.success_rate = (frame.success_rate * (frame.usage_count - 1) as f64 + 1.0) / frame.usage_count as f64;
            } else {
                frame.success_rate = (frame.success_rate * (frame.usage_count - 1) as f64) / frame.usage_count as f64;
            }
        }
    }

    fn count_considered_frames(&self, domain: &str) -> u32 {
        self.frame_database.get_frames_for_domain(domain).len() as u32
    }

    fn estimate_navigation_steps(&self, bmd_effectiveness: f64) -> u32 {
        // Higher BMD effectiveness means fewer navigation steps needed
        let base_steps = 10;
        let efficiency_reduction = (bmd_effectiveness * 8.0) as u32;
        (base_steps - efficiency_reduction).max(1)
    }

    fn create_default_frame(&self, domain: &str, user_expertise: f64) -> CognitiveFrame {
        CognitiveFrame {
            id: format!("default_{}", domain),
            domain: domain.to_string(),
            complexity_level: user_expertise,
            effectiveness_score: 0.7,
            usage_count: 0,
            success_rate: 0.8,
            frame_content: FrameContent {
                approach: "adaptive".to_string(),
                key_concepts: vec!["general".to_string()],
                explanation_style: "clear".to_string(),
                example_types: vec!["practical".to_string()],
                prerequisite_knowledge: vec![],
            },
        }
    }

    fn initialize_basic_frames(&mut self) {
        // Physics frames
        self.add_frame(CognitiveFrame {
            id: "physics_novice".to_string(),
            domain: "physics".to_string(),
            complexity_level: 0.2,
            effectiveness_score: 0.8,
            usage_count: 0,
            success_rate: 0.85,
            frame_content: FrameContent {
                approach: "conceptual_intuitive".to_string(),
                key_concepts: vec!["everyday_analogies".to_string(), "visual_models".to_string()],
                explanation_style: "simple_clear".to_string(),
                example_types: vec!["daily_life".to_string(), "visual".to_string()],
                prerequisite_knowledge: vec![],
            },
        });

        self.add_frame(CognitiveFrame {
            id: "physics_expert".to_string(),
            domain: "physics".to_string(),
            complexity_level: 0.9,
            effectiveness_score: 0.95,
            usage_count: 0,
            success_rate: 0.92,
            frame_content: FrameContent {
                approach: "mathematical_rigorous".to_string(),
                key_concepts: vec!["equations".to_string(), "derivations".to_string(), "principles".to_string()],
                explanation_style: "technical_precise".to_string(),
                example_types: vec!["mathematical".to_string(), "theoretical".to_string()],
                prerequisite_knowledge: vec!["calculus".to_string(), "linear_algebra".to_string()],
            },
        });

        // Programming frames
        self.add_frame(CognitiveFrame {
            id: "programming_beginner".to_string(),
            domain: "programming".to_string(),
            complexity_level: 0.3,
            effectiveness_score: 0.82,
            usage_count: 0,
            success_rate: 0.88,
            frame_content: FrameContent {
                approach: "step_by_step_practical".to_string(),
                key_concepts: vec!["basic_syntax".to_string(), "simple_examples".to_string()],
                explanation_style: "tutorial_guided".to_string(),
                example_types: vec!["code_snippets".to_string(), "exercises".to_string()],
                prerequisite_knowledge: vec!["basic_computer_use".to_string()],
            },
        });

        info!("Initialized {} cognitive frames", self.frame_database.frames.len());
    }

    fn add_frame(&mut self, frame: CognitiveFrame) {
        let domain = frame.domain.clone();
        let frame_id = frame.id.clone();
        
        self.frame_database.frames.insert(frame_id.clone(), frame);
        self.frame_database.domain_mappings
            .entry(domain)
            .or_insert_with(Vec::new)
            .push(frame_id);
    }
}

impl FrameDatabase {
    fn new() -> Self {
        Self {
            frames: HashMap::new(),
            domain_mappings: HashMap::new(),
        }
    }

    fn get_frames_for_domain(&self, domain: &str) -> Vec<&CognitiveFrame> {
        if let Some(frame_ids) = self.domain_mappings.get(domain) {
            frame_ids.iter()
                .filter_map(|id| self.frames.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }
}
