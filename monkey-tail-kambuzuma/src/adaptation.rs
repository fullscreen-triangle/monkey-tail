use crate::*;
use anyhow::Result;
use tracing::debug;

/// Response adapter for personalizing output based on user characteristics
pub struct ResponseAdapter {
    adaptation_history: Vec<AdaptationRecord>,
    style_templates: StyleTemplates,
}

#[derive(Debug, Clone)]
pub struct AdaptationRecord {
    pub user_expertise: f64,
    pub communication_style: CommunicationStyle,
    pub adaptation_success: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct StyleTemplates {
    pub technical_templates: Vec<String>,
    pub creative_templates: Vec<String>,
    pub direct_templates: Vec<String>,
    pub detailed_templates: Vec<String>,
}

impl ResponseAdapter {
    pub fn new() -> Self {
        Self {
            adaptation_history: Vec::new(),
            style_templates: StyleTemplates::new(),
        }
    }

    pub async fn adapt_response_for_user(
        &mut self,
        bmd_result: &BmdProcessingResult,
        identity: &SemanticIdentity,
        interaction_data: &InteractionData,
    ) -> Result<ResponseAdaptation> {
        debug!("Adapting response for user expertise level: {:.2}", 
               interaction_data.user_expertise_level);

        // Determine technical depth based on expertise and preferences
        let technical_depth = self.determine_technical_depth(identity, interaction_data);

        // Adapt content based on communication style
        let adapted_content = self.adapt_content_style(
            &bmd_result.content,
            &identity.communication_patterns.communication_style,
            &technical_depth,
        ).await?;

        // Generate interaction suggestions
        let interaction_suggestions = self.generate_interaction_suggestions(identity, interaction_data);

        // Generate follow-up questions
        let follow_up_questions = self.generate_follow_up_questions(identity, bmd_result);

        // Estimate comprehension based on adaptation
        let estimated_comprehension = self.estimate_comprehension(identity, interaction_data, &technical_depth);

        // Record adaptation for learning
        self.adaptation_history.push(AdaptationRecord {
            user_expertise: interaction_data.user_expertise_level,
            communication_style: identity.communication_patterns.communication_style.clone(),
            adaptation_success: estimated_comprehension,
            timestamp: chrono::Utc::now(),
        });

        Ok(ResponseAdaptation {
            technical_depth,
            adapted_content,
            interaction_suggestions,
            follow_up_questions,
            estimated_comprehension,
        })
    }

    fn determine_technical_depth(
        &self,
        identity: &SemanticIdentity,
        interaction_data: &InteractionData,
    ) -> TechnicalDepth {
        let expertise_level = interaction_data.user_expertise_level;
        let preferred_depth = &interaction_data.communication_preferences.technical_depth;
        let detail_preference = &identity.communication_patterns.preferred_detail_level;

        // Combine user expertise with preferences
        match (expertise_level, preferred_depth, detail_preference) {
            (level, TechnicalDepth::Expert, _) if level > 0.8 => TechnicalDepth::Expert,
            (level, TechnicalDepth::Advanced, DetailLevel::Expert) if level > 0.6 => TechnicalDepth::Advanced,
            (level, _, DetailLevel::Comprehensive) if level > 0.5 => TechnicalDepth::Advanced,
            (level, _, DetailLevel::Moderate) if level > 0.3 => TechnicalDepth::Intermediate,
            _ => TechnicalDepth::Novice,
        }
    }

    async fn adapt_content_style(
        &self,
        content: &str,
        style: &CommunicationStyle,
        technical_depth: &TechnicalDepth,
    ) -> Result<String> {
        let base_content = content.to_string();
        
        match style {
            CommunicationStyle::Technical => self.format_technical_response(&base_content, technical_depth),
            CommunicationStyle::Creative => self.format_creative_response(&base_content, technical_depth),
            CommunicationStyle::Interactive => self.format_interactive_response(&base_content, technical_depth),
            CommunicationStyle::Direct => self.format_direct_response(&base_content, technical_depth),
            CommunicationStyle::Detailed => self.format_detailed_response(&base_content, technical_depth),
        }
    }

    fn format_technical_response(&self, content: &str, depth: &TechnicalDepth) -> Result<String> {
        let prefix = match depth {
            TechnicalDepth::Expert => "Technical Analysis:",
            TechnicalDepth::Advanced => "Technical Overview:",
            TechnicalDepth::Intermediate => "Technical Explanation:",
            TechnicalDepth::Novice => "Introduction:",
        };

        Ok(format!("{}\n\n{}\n\nTechnical Details:\n- Precision level: {:?}\n- Assumes background knowledge appropriate for {:?} level", 
                   prefix, content, depth, depth))
    }

    fn format_creative_response(&self, content: &str, depth: &TechnicalDepth) -> Result<String> {
        let analogy = match depth {
            TechnicalDepth::Expert => "Like a master craftsman selecting the perfect tool...",
            TechnicalDepth::Advanced => "Imagine this concept as a well-orchestrated symphony...",
            TechnicalDepth::Intermediate => "Think of this like building with LEGO blocks...",
            TechnicalDepth::Novice => "Picture this as a simple story...",
        };

        Ok(format!("🎨 Creative Perspective:\n\n{}\n\n{}\n\n💡 Key Insight: The beauty lies in understanding how the pieces fit together!", 
                   analogy, content))
    }

    fn format_interactive_response(&self, content: &str, depth: &TechnicalDepth) -> Result<String> {
        let question_style = match depth {
            TechnicalDepth::Expert => "What specific aspects would you like to explore further?",
            TechnicalDepth::Advanced => "Which part interests you most?",
            TechnicalDepth::Intermediate => "Does this make sense so far?",
            TechnicalDepth::Novice => "Would you like me to explain any part differently?",
        };

        Ok(format!("💬 Interactive Explanation:\n\n{}\n\n🤔 {}\n\n📝 Feel free to ask follow-up questions!", 
                   content, question_style))
    }

    fn format_direct_response(&self, content: &str, _depth: &TechnicalDepth) -> Result<String> {
        Ok(format!("📋 Direct Answer:\n\n{}\n\n✅ Summary: Clear, concise information provided.", content))
    }

    fn format_detailed_response(&self, content: &str, depth: &TechnicalDepth) -> Result<String> {
        let detail_level = match depth {
            TechnicalDepth::Expert => "Comprehensive technical analysis with full mathematical rigor",
            TechnicalDepth::Advanced => "Detailed explanation with supporting theory",
            TechnicalDepth::Intermediate => "Thorough explanation with examples",
            TechnicalDepth::Novice => "Step-by-step breakdown with simple examples",
        };

        Ok(format!("📚 Detailed Explanation ({}):\n\n{}\n\n🔍 Additional Context:\nThis response is tailored for {} understanding level.", 
                   detail_level, content, depth_to_string(depth)))
    }

    fn generate_interaction_suggestions(
        &self,
        identity: &SemanticIdentity,
        interaction_data: &InteractionData,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        let expertise_level = interaction_data.user_expertise_level;
        let domain = &interaction_data.domain_context;

        if expertise_level < 0.3 {
            suggestions.push(format!("Consider starting with basic {} concepts", domain));
            suggestions.push("Try asking for examples or analogies".to_string());
        } else if expertise_level < 0.7 {
            suggestions.push(format!("Explore intermediate {} topics", domain));
            suggestions.push("Ask about practical applications".to_string());
        } else {
            suggestions.push(format!("Dive into advanced {} theory", domain));
            suggestions.push("Explore cutting-edge research in this area".to_string());
        }

        // Add style-specific suggestions
        match identity.communication_patterns.communication_style {
            CommunicationStyle::Interactive => {
                suggestions.push("Feel free to interrupt with questions".to_string());
            },
            CommunicationStyle::Creative => {
                suggestions.push("Ask for creative analogies or metaphors".to_string());
            },
            CommunicationStyle::Technical => {
                suggestions.push("Request mathematical derivations or proofs".to_string());
            },
            _ => {},
        }

        suggestions
    }

    fn generate_follow_up_questions(
        &self,
        identity: &SemanticIdentity,
        bmd_result: &BmdProcessingResult,
    ) -> Vec<String> {
        let mut questions = Vec::new();
        
        let frame = &bmd_result.selected_frame;
        
        // Generate questions based on frame content
        if !frame.frame_content.prerequisite_knowledge.is_empty() {
            questions.push("Would you like me to explain any prerequisite concepts?".to_string());
        }

        if !frame.frame_content.key_concepts.is_empty() {
            questions.push(format!("Which of these concepts would you like to explore: {}?", 
                                 frame.frame_content.key_concepts.join(", ")));
        }

        // Add style-specific questions
        match identity.communication_patterns.communication_style {
            CommunicationStyle::Interactive => {
                questions.push("What questions do you have about this?".to_string());
            },
            CommunicationStyle::Detailed => {
                questions.push("Would you like more detailed information on any aspect?".to_string());
            },
            CommunicationStyle::Creative => {
                questions.push("Would you like to see this from a different perspective?".to_string());
            },
            _ => {
                questions.push("Is there anything specific you'd like to know more about?".to_string());
            },
        }

        questions
    }

    fn estimate_comprehension(
        &self,
        identity: &SemanticIdentity,
        interaction_data: &InteractionData,
        technical_depth: &TechnicalDepth,
    ) -> f64 {
        let expertise_match = match (interaction_data.user_expertise_level, technical_depth) {
            (level, TechnicalDepth::Expert) if level > 0.8 => 0.95,
            (level, TechnicalDepth::Advanced) if level > 0.6 => 0.90,
            (level, TechnicalDepth::Intermediate) if level > 0.3 => 0.85,
            (level, TechnicalDepth::Novice) if level < 0.5 => 0.80,
            _ => 0.70, // Mismatch between expertise and depth
        };

        let style_match = match identity.communication_patterns.communication_style {
            CommunicationStyle::Technical => 0.9,
            CommunicationStyle::Direct => 0.85,
            CommunicationStyle::Detailed => 0.88,
            CommunicationStyle::Interactive => 0.82,
            CommunicationStyle::Creative => 0.80,
        };

        let complexity_adjustment = 1.0 - (interaction_data.query_complexity * 0.1);

        (expertise_match * style_match * complexity_adjustment).min(1.0).max(0.5)
    }
}

impl StyleTemplates {
    fn new() -> Self {
        Self {
            technical_templates: vec![
                "Technical Analysis:".to_string(),
                "Formal Definition:".to_string(),
                "Mathematical Framework:".to_string(),
            ],
            creative_templates: vec![
                "🎨 Creative Perspective:".to_string(),
                "💡 Imaginative Approach:".to_string(),
                "🌟 Unique Angle:".to_string(),
            ],
            direct_templates: vec![
                "📋 Direct Answer:".to_string(),
                "✅ Bottom Line:".to_string(),
                "🎯 Key Point:".to_string(),
            ],
            detailed_templates: vec![
                "📚 Comprehensive Explanation:".to_string(),
                "🔍 In-Depth Analysis:".to_string(),
                "📖 Detailed Breakdown:".to_string(),
            ],
        }
    }
}

/// Learning analyzer for extracting insights from interactions
pub struct LearningAnalyzer {
    insight_history: Vec<LearningInsight>,
    pattern_detector: PatternDetector,
}

#[derive(Debug, Clone)]
pub struct PatternDetector {
    knowledge_gap_patterns: Vec<String>,
    strength_indicators: Vec<String>,
    misconception_signals: Vec<String>,
}

impl LearningAnalyzer {
    pub fn new() -> Self {
        Self {
            insight_history: Vec::new(),
            pattern_detector: PatternDetector::new(),
        }
    }

    pub async fn extract_learning_insights(
        &mut self,
        query: &str,
        bmd_result: &BmdProcessingResult,
        identity: &SemanticIdentity,
    ) -> Result<Vec<LearningInsight>> {
        let mut insights = Vec::new();

        // Analyze query for knowledge gaps
        if let Some(gap_insight) = self.detect_knowledge_gaps(query, identity).await? {
            insights.push(gap_insight);
        }

        // Analyze for learning opportunities
        if let Some(opportunity_insight) = self.detect_learning_opportunities(query, bmd_result).await? {
            insights.push(opportunity_insight);
        }

        // Analyze for conceptual connections
        if let Some(connection_insight) = self.detect_conceptual_connections(query, identity).await? {
            insights.push(connection_insight);
        }

        // Store insights for pattern learning
        self.insight_history.extend(insights.clone());

        Ok(insights)
    }

    async fn detect_knowledge_gaps(
        &self,
        query: &str,
        identity: &SemanticIdentity,
    ) -> Result<Option<LearningInsight>> {
        // Simple pattern matching for knowledge gaps
        let gap_indicators = ["what is", "how does", "explain", "don't understand"];
        
        if gap_indicators.iter().any(|indicator| query.to_lowercase().contains(indicator)) {
            return Ok(Some(LearningInsight {
                domain: "general".to_string(),
                insight_type: InsightType::KnowledgeGap,
                confidence: 0.7,
                description: "Query indicates potential knowledge gap in fundamental concepts".to_string(),
                suggested_next_steps: vec![
                    "Start with basic definitions".to_string(),
                    "Provide concrete examples".to_string(),
                    "Build up from familiar concepts".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    async fn detect_learning_opportunities(
        &self,
        query: &str,
        bmd_result: &BmdProcessingResult,
    ) -> Result<Option<LearningInsight>> {
        let frame = &bmd_result.selected_frame;
        
        if !frame.frame_content.prerequisite_knowledge.is_empty() {
            return Ok(Some(LearningInsight {
                domain: frame.domain.clone(),
                insight_type: InsightType::LearningOpportunity,
                confidence: 0.8,
                description: format!("Opportunity to strengthen foundation in {}", frame.domain),
                suggested_next_steps: frame.frame_content.prerequisite_knowledge.iter()
                    .map(|prereq| format!("Review {}", prereq))
                    .collect(),
            }));
        }

        Ok(None)
    }

    async fn detect_conceptual_connections(
        &self,
        query: &str,
        identity: &SemanticIdentity,
    ) -> Result<Option<LearningInsight>> {
        // Look for opportunities to connect concepts across domains
        let domains: Vec<_> = identity.understanding_vector.domains.keys().collect();
        
        if domains.len() > 1 {
            return Ok(Some(LearningInsight {
                domain: "interdisciplinary".to_string(),
                insight_type: InsightType::ConceptualConnection,
                confidence: 0.6,
                description: "Opportunity to connect concepts across multiple domains".to_string(),
                suggested_next_steps: vec![
                    format!("Explore connections between {}", domains.join(" and ")),
                    "Look for common patterns and principles".to_string(),
                ],
            }));
        }

        Ok(None)
    }
}

impl PatternDetector {
    fn new() -> Self {
        Self {
            knowledge_gap_patterns: vec![
                "what is".to_string(),
                "how does".to_string(),
                "explain".to_string(),
                "don't understand".to_string(),
                "confused about".to_string(),
            ],
            strength_indicators: vec![
                "already know".to_string(),
                "familiar with".to_string(),
                "understand that".to_string(),
                "can you elaborate".to_string(),
            ],
            misconception_signals: vec![
                "I thought".to_string(),
                "isn't it true that".to_string(),
                "but I heard".to_string(),
            ],
        }
    }
}

fn depth_to_string(depth: &TechnicalDepth) -> &'static str {
    match depth {
        TechnicalDepth::Novice => "novice",
        TechnicalDepth::Intermediate => "intermediate",
        TechnicalDepth::Advanced => "advanced",
        TechnicalDepth::Expert => "expert",
    }
}
