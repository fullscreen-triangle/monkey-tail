use crate::*;
use anyhow::Result;
use std::collections::VecDeque;
use tracing::debug;

/// Pattern analyzer for understanding user behavioral patterns
pub struct PatternAnalyzer {
    interaction_history: VecDeque<InteractionRecord>,
    pattern_cache: HashMap<Uuid, UserPatterns>,
    max_history_size: usize,
}

#[derive(Debug, Clone)]
pub struct InteractionRecord {
    pub user_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub query: String,
    pub query_length: usize,
    pub query_complexity: f64,
    pub domain: String,
    pub processing_time_ms: u64,
    pub bmd_effectiveness: f64,
    pub confidence_score: f64,
}

#[derive(Debug, Clone)]
pub struct UserPatterns {
    pub typing_patterns: TypingPatterns,
    pub temporal_patterns: UserTemporalPatterns,
    pub cognitive_patterns: CognitivePatterns,
    pub interaction_patterns: UserInteractionPatterns,
    pub learning_patterns: UserLearningPatterns,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct TypingPatterns {
    pub average_query_length: f64,
    pub query_length_variance: f64,
    pub typing_rhythm: Vec<f64>, // Time between characters (estimated)
    pub punctuation_usage: HashMap<char, f64>,
    pub vocabulary_complexity: f64,
}

#[derive(Debug, Clone)]
pub struct UserTemporalPatterns {
    pub active_hours: HashMap<u8, f64>, // Hour -> activity level
    pub session_durations: Vec<f64>, // Minutes
    pub break_patterns: Vec<f64>, // Minutes between sessions
    pub day_of_week_patterns: HashMap<String, f64>,
    pub response_time_patterns: Vec<f64>, // Seconds between queries
}

#[derive(Debug, Clone)]
pub struct CognitivePatterns {
    pub complexity_preference: f64,
    pub domain_switching_frequency: f64,
    pub depth_vs_breadth_preference: f64, // 0.0 = breadth, 1.0 = depth
    pub abstract_vs_concrete_preference: f64, // 0.0 = concrete, 1.0 = abstract
    pub learning_velocity_by_domain: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct UserInteractionPatterns {
    pub question_types: HashMap<String, u32>, // "what", "how", "why", etc.
    pub follow_up_frequency: f64,
    pub clarification_requests: f64,
    pub example_requests: f64,
    pub interruption_patterns: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct UserLearningPatterns {
    pub repetition_needs: HashMap<String, f64>, // Domain -> repetition frequency
    pub mistake_patterns: Vec<String>,
    pub breakthrough_indicators: Vec<String>,
    pub plateau_detection: HashMap<String, f64>, // Domain -> plateau probability
    pub learning_style_indicators: LearningStyleIndicators,
}

#[derive(Debug, Clone)]
pub struct LearningStyleIndicators {
    pub visual_learning_score: f64,
    pub auditory_learning_score: f64,
    pub kinesthetic_learning_score: f64,
    pub reading_learning_score: f64,
    pub social_learning_score: f64,
    pub solitary_learning_score: f64,
}

impl PatternAnalyzer {
    pub fn new() -> Self {
        Self {
            interaction_history: VecDeque::new(),
            pattern_cache: HashMap::new(),
            max_history_size: 1000, // Keep last 1000 interactions
        }
    }

    /// Analyze interaction patterns for a user
    pub async fn analyze_interaction_patterns(
        &mut self,
        user_id: Uuid,
        query: &str,
        result: &PersonalizedProcessingResult,
    ) -> Result<()> {
        debug!("Analyzing interaction patterns for user {}", user_id);

        // Record the interaction
        let record = InteractionRecord {
            user_id,
            timestamp: Utc::now(),
            query: query.to_string(),
            query_length: query.len(),
            query_complexity: self.estimate_query_complexity(query),
            domain: self.extract_domain_from_metadata(&result.metadata),
            processing_time_ms: result.processing_time_ms,
            bmd_effectiveness: result.bmd_effectiveness,
            confidence_score: result.confidence_score,
        };

        // Add to history (maintain size limit)
        self.interaction_history.push_back(record);
        if self.interaction_history.len() > self.max_history_size {
            self.interaction_history.pop_front();
        }

        // Update patterns for this user
        self.update_user_patterns(user_id).await?;

        Ok(())
    }

    async fn update_user_patterns(&mut self, user_id: Uuid) -> Result<()> {
        let user_interactions: Vec<_> = self.interaction_history
            .iter()
            .filter(|record| record.user_id == user_id)
            .collect();

        if user_interactions.is_empty() {
            return Ok(());
        }

        let patterns = UserPatterns {
            typing_patterns: self.analyze_typing_patterns(&user_interactions)?,
            temporal_patterns: self.analyze_temporal_patterns(&user_interactions)?,
            cognitive_patterns: self.analyze_cognitive_patterns(&user_interactions)?,
            interaction_patterns: self.analyze_user_interaction_patterns(&user_interactions)?,
            learning_patterns: self.analyze_learning_patterns(&user_interactions)?,
            last_updated: Utc::now(),
        };

        self.pattern_cache.insert(user_id, patterns);
        Ok(())
    }

    fn analyze_typing_patterns(&self, interactions: &[&InteractionRecord]) -> Result<TypingPatterns> {
        let query_lengths: Vec<f64> = interactions.iter().map(|i| i.query_length as f64).collect();
        
        let average_query_length = query_lengths.iter().sum::<f64>() / query_lengths.len() as f64;
        
        let query_length_variance = if query_lengths.len() > 1 {
            let mean = average_query_length;
            query_lengths.iter()
                .map(|&length| (length - mean).powi(2))
                .sum::<f64>() / query_lengths.len() as f64
        } else {
            0.0
        };

        // Analyze vocabulary complexity
        let total_words: usize = interactions.iter()
            .map(|i| i.query.split_whitespace().count())
            .sum();
        let unique_words: std::collections::HashSet<_> = interactions.iter()
            .flat_map(|i| i.query.split_whitespace())
            .collect();
        
        let vocabulary_complexity = if total_words > 0 {
            unique_words.len() as f64 / total_words as f64
        } else {
            0.0
        };

        // Analyze punctuation usage
        let mut punctuation_usage = HashMap::new();
        let total_chars: usize = interactions.iter().map(|i| i.query.len()).sum();
        
        for interaction in interactions {
            for ch in interaction.query.chars() {
                if ch.is_ascii_punctuation() {
                    *punctuation_usage.entry(ch).or_insert(0.0) += 1.0;
                }
            }
        }
        
        // Normalize punctuation usage
        for count in punctuation_usage.values_mut() {
            *count /= total_chars as f64;
        }

        Ok(TypingPatterns {
            average_query_length,
            query_length_variance,
            typing_rhythm: vec![], // Would need keystroke timing data
            punctuation_usage,
            vocabulary_complexity,
        })
    }

    fn analyze_temporal_patterns(&self, interactions: &[&InteractionRecord]) -> Result<UserTemporalPatterns> {
        let mut active_hours = HashMap::new();
        let mut response_times = Vec::new();
        
        // Analyze activity by hour
        for interaction in interactions {
            let hour = interaction.timestamp.hour() as u8;
            *active_hours.entry(hour).or_insert(0.0) += 1.0;
        }

        // Calculate response times between queries
        for window in interactions.windows(2) {
            let time_diff = window[1].timestamp.signed_duration_since(window[0].timestamp);
            response_times.push(time_diff.num_seconds() as f64);
        }

        // Analyze day of week patterns
        let mut day_patterns = HashMap::new();
        for interaction in interactions {
            let day = interaction.timestamp.format("%A").to_string();
            *day_patterns.entry(day).or_insert(0.0) += 1.0;
        }

        Ok(UserTemporalPatterns {
            active_hours,
            session_durations: vec![], // Would need session boundary detection
            break_patterns: vec![], // Would need session boundary detection
            day_of_week_patterns: day_patterns,
            response_time_patterns: response_times,
        })
    }

    fn analyze_cognitive_patterns(&self, interactions: &[&InteractionRecord]) -> Result<CognitivePatterns> {
        // Calculate complexity preference
        let complexity_preference = interactions.iter()
            .map(|i| i.query_complexity)
            .sum::<f64>() / interactions.len() as f64;

        // Analyze domain switching
        let domains: Vec<_> = interactions.iter().map(|i| &i.domain).collect();
        let unique_domains: std::collections::HashSet<_> = domains.iter().collect();
        let domain_switches = domains.windows(2)
            .filter(|window| window[0] != window[1])
            .count();
        
        let domain_switching_frequency = if interactions.len() > 1 {
            domain_switches as f64 / (interactions.len() - 1) as f64
        } else {
            0.0
        };

        // Calculate learning velocity by domain
        let mut learning_velocity_by_domain = HashMap::new();
        for domain in unique_domains {
            let domain_interactions: Vec<_> = interactions.iter()
                .filter(|i| &i.domain == *domain)
                .collect();
            
            if domain_interactions.len() > 1 {
                let first_bmd = domain_interactions.first().unwrap().bmd_effectiveness;
                let last_bmd = domain_interactions.last().unwrap().bmd_effectiveness;
                let velocity = (last_bmd - first_bmd) / domain_interactions.len() as f64;
                learning_velocity_by_domain.insert(domain.to_string(), velocity);
            }
        }

        Ok(CognitivePatterns {
            complexity_preference,
            domain_switching_frequency,
            depth_vs_breadth_preference: 0.5, // Would need more sophisticated analysis
            abstract_vs_concrete_preference: 0.5, // Would need content analysis
            learning_velocity_by_domain,
        })
    }

    fn analyze_user_interaction_patterns(&self, interactions: &[&InteractionRecord]) -> Result<UserInteractionPatterns> {
        let mut question_types = HashMap::new();
        
        // Analyze question types
        for interaction in interactions {
            let query_lower = interaction.query.to_lowercase();
            
            if query_lower.starts_with("what") {
                *question_types.entry("what".to_string()).or_insert(0) += 1;
            } else if query_lower.starts_with("how") {
                *question_types.entry("how".to_string()).or_insert(0) += 1;
            } else if query_lower.starts_with("why") {
                *question_types.entry("why".to_string()).or_insert(0) += 1;
            } else if query_lower.starts_with("when") {
                *question_types.entry("when".to_string()).or_insert(0) += 1;
            } else if query_lower.starts_with("where") {
                *question_types.entry("where".to_string()).or_insert(0) += 1;
            } else {
                *question_types.entry("other".to_string()).or_insert(0) += 1;
            }
        }

        // Calculate follow-up frequency (simplified)
        let follow_up_indicators = ["also", "additionally", "furthermore", "moreover"];
        let follow_ups = interactions.iter()
            .filter(|i| follow_up_indicators.iter().any(|&indicator| i.query.to_lowercase().contains(indicator)))
            .count();
        
        let follow_up_frequency = follow_ups as f64 / interactions.len() as f64;

        // Calculate clarification requests
        let clarification_indicators = ["clarify", "explain", "what do you mean", "i don't understand"];
        let clarifications = interactions.iter()
            .filter(|i| clarification_indicators.iter().any(|&indicator| i.query.to_lowercase().contains(indicator)))
            .count();
        
        let clarification_requests = clarifications as f64 / interactions.len() as f64;

        // Calculate example requests
        let example_indicators = ["example", "for instance", "show me", "demonstrate"];
        let examples = interactions.iter()
            .filter(|i| example_indicators.iter().any(|&indicator| i.query.to_lowercase().contains(indicator)))
            .count();
        
        let example_requests = examples as f64 / interactions.len() as f64;

        Ok(UserInteractionPatterns {
            question_types,
            follow_up_frequency,
            clarification_requests,
            example_requests,
            interruption_patterns: vec![], // Would need real-time interaction data
        })
    }

    fn analyze_learning_patterns(&self, interactions: &[&InteractionRecord]) -> Result<UserLearningPatterns> {
        // Analyze repetition needs by domain
        let mut repetition_needs = HashMap::new();
        let mut domain_queries: HashMap<String, Vec<&InteractionRecord>> = HashMap::new();
        
        for interaction in interactions {
            domain_queries.entry(interaction.domain.clone())
                .or_insert_with(Vec::new)
                .push(interaction);
        }

        for (domain, queries) in domain_queries {
            if queries.len() > 1 {
                // Calculate how often similar queries repeat in this domain
                let repetition_score = self.calculate_repetition_score(&queries);
                repetition_needs.insert(domain, repetition_score);
            }
        }

        // Detect learning style indicators
        let learning_style_indicators = self.detect_learning_style_indicators(interactions)?;

        Ok(UserLearningPatterns {
            repetition_needs,
            mistake_patterns: vec![], // Would need error detection
            breakthrough_indicators: vec![], // Would need success pattern detection
            plateau_detection: HashMap::new(), // Would need longitudinal analysis
            learning_style_indicators,
        })
    }

    fn calculate_repetition_score(&self, queries: &[&InteractionRecord]) -> f64 {
        // Simple similarity-based repetition detection
        let mut repetition_count = 0;
        let total_comparisons = queries.len() * (queries.len() - 1) / 2;
        
        for i in 0..queries.len() {
            for j in (i + 1)..queries.len() {
                let similarity = self.calculate_query_similarity(&queries[i].query, &queries[j].query);
                if similarity > 0.7 { // Threshold for considering queries similar
                    repetition_count += 1;
                }
            }
        }

        if total_comparisons > 0 {
            repetition_count as f64 / total_comparisons as f64
        } else {
            0.0
        }
    }

    fn calculate_query_similarity(&self, query1: &str, query2: &str) -> f64 {
        // Simple word-based similarity
        let words1: std::collections::HashSet<_> = query1.split_whitespace().collect();
        let words2: std::collections::HashSet<_> = query2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    fn detect_learning_style_indicators(&self, interactions: &[&InteractionRecord]) -> Result<LearningStyleIndicators> {
        let total_interactions = interactions.len() as f64;
        
        // Visual learning indicators
        let visual_indicators = ["show", "diagram", "chart", "picture", "visual", "see"];
        let visual_count = interactions.iter()
            .filter(|i| visual_indicators.iter().any(|&indicator| i.query.to_lowercase().contains(indicator)))
            .count() as f64;
        
        // Reading learning indicators
        let reading_indicators = ["read", "text", "document", "article", "book"];
        let reading_count = interactions.iter()
            .filter(|i| reading_indicators.iter().any(|&indicator| i.query.to_lowercase().contains(indicator)))
            .count() as f64;
        
        // Kinesthetic learning indicators
        let kinesthetic_indicators = ["hands-on", "practice", "try", "do", "exercise"];
        let kinesthetic_count = interactions.iter()
            .filter(|i| kinesthetic_indicators.iter().any(|&indicator| i.query.to_lowercase().contains(indicator)))
            .count() as f64;

        Ok(LearningStyleIndicators {
            visual_learning_score: visual_count / total_interactions,
            auditory_learning_score: 0.5, // Would need audio interaction data
            kinesthetic_learning_score: kinesthetic_count / total_interactions,
            reading_learning_score: reading_count / total_interactions,
            social_learning_score: 0.5, // Would need social interaction data
            solitary_learning_score: 0.5, // Default assumption
        })
    }

    fn estimate_query_complexity(&self, query: &str) -> f64 {
        let word_count = query.split_whitespace().count() as f64;
        let char_count = query.len() as f64;
        let question_words = ["what", "how", "why", "when", "where", "which"];
        let has_question_word = question_words.iter().any(|&word| query.to_lowercase().contains(word));
        
        // Simple complexity estimation
        let length_factor = (word_count / 10.0).min(1.0);
        let complexity_factor = if has_question_word { 0.7 } else { 0.5 };
        let char_factor = (char_count / 100.0).min(1.0);
        
        (length_factor + complexity_factor + char_factor) / 3.0
    }

    fn extract_domain_from_metadata(&self, metadata: &ProcessingMetadata) -> String {
        // Extract the most prominent domain from detected expertise
        metadata.domain_expertise_detected
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(domain, _)| domain.clone())
            .unwrap_or_else(|| "general".to_string())
    }

    /// Get patterns for a specific user
    pub fn get_user_patterns(&self, user_id: Uuid) -> Option<&UserPatterns> {
        self.pattern_cache.get(&user_id)
    }

    /// Get interaction history for a user
    pub fn get_user_interaction_history(&self, user_id: Uuid) -> Vec<&InteractionRecord> {
        self.interaction_history
            .iter()
            .filter(|record| record.user_id == user_id)
            .collect()
    }

    /// Clear patterns for a user (privacy)
    pub fn clear_user_patterns(&mut self, user_id: Uuid) {
        self.pattern_cache.remove(&user_id);
        self.interaction_history.retain(|record| record.user_id != user_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use monkey_tail_kambuzuma::{ResponseAdaptation, TechnicalDepth, ProcessingMetadata};

    #[tokio::test]
    async fn test_pattern_analyzer_creation() {
        let analyzer = PatternAnalyzer::new();
        assert_eq!(analyzer.max_history_size, 1000);
        assert!(analyzer.interaction_history.is_empty());
        assert!(analyzer.pattern_cache.is_empty());
    }

    #[tokio::test]
    async fn test_interaction_pattern_analysis() {
        let mut analyzer = PatternAnalyzer::new();
        let user_id = Uuid::new_v4();
        
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
            learning_insights: vec![],
            confidence_score: 0.9,
            processing_time_ms: 150,
            metadata: ProcessingMetadata {
                bmd_frames_considered: 3,
                s_entropy_navigation_steps: 5,
                adaptation_iterations: 1,
                domain_expertise_detected: {
                    let mut map = HashMap::new();
                    map.insert("test".to_string(), 0.7);
                    map
                },
            },
        };

        // Analyze interaction
        analyzer.analyze_interaction_patterns(user_id, "What is machine learning?", &result).await.unwrap();
        
        // Verify interaction was recorded
        assert_eq!(analyzer.interaction_history.len(), 1);
        assert_eq!(analyzer.interaction_history[0].user_id, user_id);
        assert_eq!(analyzer.interaction_history[0].query, "What is machine learning?");
    }

    #[tokio::test]
    async fn test_user_pattern_extraction() {
        let mut analyzer = PatternAnalyzer::new();
        let user_id = Uuid::new_v4();
        
        // Add multiple interactions
        for i in 0..5 {
            let result = PersonalizedProcessingResult {
                content: format!("Response {}", i),
                bmd_effectiveness: 0.7 + (i as f64 * 0.05),
                response_adaptation: ResponseAdaptation {
                    technical_depth: TechnicalDepth::Intermediate,
                    adapted_content: format!("Adapted response {}", i),
                    interaction_suggestions: vec![],
                    follow_up_questions: vec![],
                    estimated_comprehension: 0.8,
                },
                learning_insights: vec![],
                confidence_score: 0.85,
                processing_time_ms: 100 + (i * 10),
                metadata: ProcessingMetadata {
                    bmd_frames_considered: 3,
                    s_entropy_navigation_steps: 5,
                    adaptation_iterations: 1,
                    domain_expertise_detected: {
                        let mut map = HashMap::new();
                        map.insert("test".to_string(), 0.6 + (i as f64 * 0.1));
                        map
                    },
                },
            };

            analyzer.analyze_interaction_patterns(
                user_id, 
                &format!("What is test question {}?", i), 
                &result
            ).await.unwrap();
        }

        // Verify patterns were extracted
        let patterns = analyzer.get_user_patterns(user_id);
        assert!(patterns.is_some());
        
        let patterns = patterns.unwrap();
        assert!(patterns.typing_patterns.average_query_length > 0.0);
        assert!(!patterns.temporal_patterns.active_hours.is_empty());
        assert!(patterns.cognitive_patterns.complexity_preference > 0.0);
    }

    #[test]
    fn test_query_similarity_calculation() {
        let analyzer = PatternAnalyzer::new();
        
        let similarity1 = analyzer.calculate_query_similarity(
            "What is machine learning?", 
            "What is machine learning?"
        );
        assert_eq!(similarity1, 1.0);
        
        let similarity2 = analyzer.calculate_query_similarity(
            "What is machine learning?", 
            "How does deep learning work?"
        );
        assert!(similarity2 < 1.0 && similarity2 > 0.0);
        
        let similarity3 = analyzer.calculate_query_similarity(
            "What is machine learning?", 
            "Tell me about cats"
        );
        assert!(similarity3 < 0.5);
    }
}
