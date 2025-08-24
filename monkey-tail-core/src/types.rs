use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Environmental context extracted from sensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalContext {
    pub timestamp: DateTime<Utc>,
    pub location_context: LocationContext,
    pub device_context: DeviceContext,
    pub temporal_context: TemporalContextData,
    pub interaction_context: InteractionContextData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationContext {
    pub environment_type: EnvironmentType,
    pub noise_level: f64,
    pub lighting_conditions: LightingConditions,
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvironmentType {
    Home,
    Office,
    Public,
    Vehicle,
    Outdoor,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightingConditions {
    Bright,
    Moderate,
    Dim,
    Dark,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceContext {
    pub device_type: DeviceType,
    pub screen_size: Option<(u32, u32)>,
    pub input_methods: Vec<InputMethod>,
    pub connectivity: ConnectivityStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    Desktop,
    Laptop,
    Tablet,
    Phone,
    Wearable,
    IoT,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputMethod {
    Keyboard,
    Mouse,
    Touch,
    Voice,
    Gesture,
    Eye,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityStatus {
    Online,
    Offline,
    Limited,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContextData {
    pub time_of_day: TimeOfDay,
    pub day_of_week: DayOfWeek,
    pub session_duration: std::time::Duration,
    pub time_since_last_interaction: Option<std::time::Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeOfDay {
    EarlyMorning,  // 5-8 AM
    Morning,       // 8-12 PM
    Afternoon,     // 12-5 PM
    Evening,       // 5-9 PM
    Night,         // 9 PM-5 AM
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DayOfWeek {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionContextData {
    pub interaction_type: InteractionType,
    pub urgency_level: UrgencyLevel,
    pub complexity_estimate: f64,
    pub expected_duration: Option<std::time::Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Query,
    Conversation,
    Task,
    Learning,
    Problem_Solving,
    Creative,
    Administrative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Ecosystem signature for security through uniqueness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemSignature {
    pub machine_fingerprint: MachineFingerprint,
    pub environment_signature: EnvironmentSignature,
    pub interaction_patterns: InteractionPatternSignature,
    pub temporal_signature: TemporalSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineFingerprint {
    pub hardware_id: String,
    pub os_signature: String,
    pub network_signature: String,
    pub performance_profile: PerformanceProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub cpu_characteristics: String,
    pub memory_profile: String,
    pub storage_profile: String,
    pub network_latency_profile: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSignature {
    pub sensor_fingerprint: String,
    pub environmental_patterns: HashMap<String, f64>,
    pub context_stability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPatternSignature {
    pub typing_patterns: TypingPattern,
    pub interaction_rhythms: Vec<f64>,
    pub preference_stability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingPattern {
    pub keystroke_dynamics: Vec<f64>,
    pub pause_patterns: Vec<f64>,
    pub error_patterns: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSignature {
    pub activity_patterns: HashMap<String, f64>,
    pub session_patterns: Vec<f64>,
    pub circadian_signature: Vec<f64>,
}

impl Default for EnvironmentalContext {
    fn default() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            location_context: LocationContext {
                environment_type: EnvironmentType::Unknown,
                noise_level: 0.5,
                lighting_conditions: LightingConditions::Unknown,
                temperature: None,
            },
            device_context: DeviceContext {
                device_type: DeviceType::Unknown,
                screen_size: None,
                input_methods: vec![InputMethod::Keyboard],
                connectivity: ConnectivityStatus::Online,
            },
            temporal_context: TemporalContextData {
                time_of_day: TimeOfDay::Morning,
                day_of_week: DayOfWeek::Monday,
                session_duration: std::time::Duration::from_secs(0),
                time_since_last_interaction: None,
            },
            interaction_context: InteractionContextData {
                interaction_type: InteractionType::Query,
                urgency_level: UrgencyLevel::Medium,
                complexity_estimate: 0.5,
                expected_duration: None,
            },
        }
    }
}

impl EcosystemSignature {
    pub fn generate_unique() -> Self {
        Self {
            machine_fingerprint: MachineFingerprint {
                hardware_id: format!("hw_{}", Uuid::new_v4()),
                os_signature: std::env::consts::OS.to_string(),
                network_signature: format!("net_{}", Uuid::new_v4()),
                performance_profile: PerformanceProfile {
                    cpu_characteristics: "generic".to_string(),
                    memory_profile: "standard".to_string(),
                    storage_profile: "ssd".to_string(),
                    network_latency_profile: vec![10.0, 15.0, 12.0],
                },
            },
            environment_signature: EnvironmentSignature {
                sensor_fingerprint: format!("sensor_{}", Uuid::new_v4()),
                environmental_patterns: HashMap::new(),
                context_stability: 0.8,
            },
            interaction_patterns: InteractionPatternSignature {
                typing_patterns: TypingPattern {
                    keystroke_dynamics: vec![0.1, 0.15, 0.12],
                    pause_patterns: vec![0.5, 0.3, 0.8],
                    error_patterns: vec![0.02, 0.01, 0.03],
                },
                interaction_rhythms: vec![1.0, 0.8, 1.2],
                preference_stability: 0.9,
            },
            temporal_signature: TemporalSignature {
                activity_patterns: HashMap::new(),
                session_patterns: vec![30.0, 45.0, 60.0],
                circadian_signature: vec![0.3, 0.8, 1.0, 0.6],
            },
        }
    }

    pub fn calculate_uniqueness(&self) -> f64 {
        // Simplified uniqueness calculation
        // In practice, this would use sophisticated fingerprinting
        let machine_uniqueness = self.machine_fingerprint.hardware_id.len() as f64 / 100.0;
        let environment_uniqueness = self.environment_signature.context_stability;
        let interaction_uniqueness = self.interaction_patterns.preference_stability;
        let temporal_uniqueness = self.temporal_signature.session_patterns.len() as f64 / 10.0;

        (machine_uniqueness * environment_uniqueness * interaction_uniqueness * temporal_uniqueness)
            .min(1.0)
    }

    pub fn hash(&self) -> String {
        // Simplified hash - in practice would use cryptographic hashing
        format!("{}_{}_{}_{}", 
            &self.machine_fingerprint.hardware_id[..8],
            &self.environment_signature.sensor_fingerprint[..8],
            self.interaction_patterns.preference_stability,
            self.temporal_signature.session_patterns.len()
        )
    }

    pub fn calculate_combined_uniqueness(&self, other: &PersonSignature) -> f64 {
        let ecosystem_uniqueness = self.calculate_uniqueness();
        let person_uniqueness = other.calculate_uniqueness();
        
        // Geometric mean for security
        (ecosystem_uniqueness * person_uniqueness).sqrt()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonSignature {
    pub behavioral_patterns: HashMap<String, f64>,
    pub interaction_style: String,
    pub preference_consistency: f64,
    pub temporal_patterns: Vec<f64>,
}

impl PersonSignature {
    pub fn calculate_uniqueness(&self) -> f64 {
        let pattern_diversity = self.behavioral_patterns.len() as f64 / 20.0; // Normalize to 20 patterns
        let consistency = self.preference_consistency;
        let temporal_complexity = self.temporal_patterns.len() as f64 / 24.0; // 24 hours
        
        (pattern_diversity * consistency * temporal_complexity).min(1.0)
    }

    pub fn hash(&self) -> String {
        format!("person_{}_{:.2}_{}", 
            self.interaction_style,
            self.preference_consistency,
            self.temporal_patterns.len()
        )
    }
}

impl Default for PersonSignature {
    fn default() -> Self {
        Self {
            behavioral_patterns: HashMap::new(),
            interaction_style: "default".to_string(),
            preference_consistency: 0.5,
            temporal_patterns: vec![0.5; 24], // 24 hours of default activity
        }
    }
}
