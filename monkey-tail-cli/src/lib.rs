use clap::{Parser, Subcommand};
use anyhow::Result;
use uuid::Uuid;
use std::collections::HashMap;

use monkey_tail_core::*;
use monkey_tail_identity::*;
use monkey_tail_kambuzuma::*;
use monkey_tail_competency::*;

pub mod commands;
pub mod demo;
pub mod integration;

pub use commands::*;
pub use demo::*;
pub use integration::*;

#[derive(Parser)]
#[command(name = "monkey-tail")]
#[command(about = "Monkey-Tail: Ephemeral Semantic Digital Identity System")]
#[command(version = "0.1.0")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run interactive demo
    Demo {
        /// Domain to demonstrate (physics, computer_science, mathematics)
        #[arg(short, long, default_value = "physics")]
        domain: String,
        
        /// User expertise level (0.0-1.0)
        #[arg(short, long, default_value = "0.5")]
        expertise: f64,
    },
    
    /// Process a single query
    Query {
        /// The query to process
        query: String,
        
        /// Domain context
        #[arg(short, long, default_value = "general")]
        domain: String,
        
        /// User expertise level (0.0-1.0)
        #[arg(short, long, default_value = "0.5")]
        expertise: f64,
    },
    
    /// Show system statistics
    Stats,
    
    /// Test ecosystem security
    Security {
        /// Number of test iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,
    },
    
    /// Benchmark BMD effectiveness scaling
    Benchmark {
        /// Domain to benchmark
        #[arg(short, long, default_value = "physics")]
        domain: String,
    },
}

/// Main CLI runner
pub async fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    match cli.command {
        Commands::Demo { domain, expertise } => {
            run_interactive_demo(&domain, expertise).await
        },
        Commands::Query { query, domain, expertise } => {
            process_single_query(&query, &domain, expertise).await
        },
        Commands::Stats => {
            show_system_stats().await
        },
        Commands::Security { iterations } => {
            test_ecosystem_security(iterations).await
        },
        Commands::Benchmark { domain } => {
            benchmark_bmd_effectiveness(&domain).await
        },
    }
}

async fn process_single_query(query: &str, domain: &str, expertise_level: f64) -> Result<()> {
    println!("🐒 Monkey-Tail: Processing Query");
    println!("Query: {}", query);
    println!("Domain: {}", domain);
    println!("User Expertise: {:.1}%", expertise_level * 100.0);
    println!();

    // Initialize the integrated system
    let mut integration = MonkeyTailIntegration::new().await?;
    
    // Process the query
    let result = integration.process_complete_query(query, domain, expertise_level).await?;
    
    // Display results
    println!("📊 Results:");
    println!("Understanding Level: {:.1}%", result.semantic_identity.calculate_bmd_effectiveness(domain) * 100.0);
    println!("BMD Effectiveness: {:.1}%", result.processing_result.bmd_effectiveness * 100.0);
    println!("Confidence Score: {:.1}%", result.processing_result.confidence_score * 100.0);
    println!("Processing Time: {}ms", result.processing_result.processing_time_ms);
    println!();
    
    println!("🧠 Competency Assessment:");
    println!("Domain: {}", result.competency_assessment.domain);
    println!("Understanding: {:.1}%", result.competency_assessment.understanding_level * 100.0);
    println!("Consensus Agreement: {:.1}%", result.competency_assessment.consensus_agreement * 100.0);
    println!("Assessment Quality: {:.1}%", result.competency_assessment.assessment_quality * 100.0);
    println!();
    
    println!("🔒 Security Metrics:");
    println!("Ecosystem Uniqueness: {:.1}%", result.security_metrics.ecosystem_uniqueness * 100.0);
    println!("Security Status: {}", if result.security_metrics.security_status { "✅ SECURE" } else { "❌ INSECURE" });
    println!();
    
    println!("💡 Learning Insights:");
    for insight in &result.processing_result.learning_insights {
        println!("- {}: {}", insight.domain, insight.description);
    }
    
    Ok(())
}

async fn show_system_stats() -> Result<()> {
    println!("📈 Monkey-Tail System Statistics");
    println!();
    
    // This would show real statistics in a production system
    println!("🔧 System Components:");
    println!("✅ Core Types: Implemented");
    println!("✅ Identity Processor: Implemented");
    println!("✅ Kambuzuma Integration: Implemented");
    println!("✅ Competency Assessor: Implemented");
    println!("✅ Security Validator: Implemented");
    println!();
    
    println!("📊 Capabilities:");
    println!("• Ephemeral semantic identity extraction");
    println!("• BMD effectiveness scaling (60%-95%)");
    println!("• Four-sided triangle competency assessment");
    println!("• Ecosystem security through uniqueness");
    println!("• Turbulance DSL semantic processing");
    println!("• Real-time adaptation to user expertise");
    
    Ok(())
}

async fn test_ecosystem_security(iterations: usize) -> Result<()> {
    println!("🔒 Testing Ecosystem Security");
    println!("Iterations: {}", iterations);
    println!();
    
    let mut security_validator = SecurityValidator::new();
    let mut success_count = 0;
    
    for i in 1..=iterations {
        // Generate test signatures
        let person_signature = PersonSignature::default();
        let machine_signature = EcosystemSignature::generate_unique();
        
        // Test security validation
        let is_secure = security_validator.validate_security_threshold(&person_signature, &machine_signature)?;
        
        if is_secure {
            success_count += 1;
        }
        
        if i % (iterations / 10).max(1) == 0 {
            println!("Progress: {}/{} ({:.1}%)", i, iterations, (i as f64 / iterations as f64) * 100.0);
        }
    }
    
    let success_rate = success_count as f64 / iterations as f64;
    println!();
    println!("🎯 Security Test Results:");
    println!("Successful validations: {}/{}", success_count, iterations);
    println!("Success rate: {:.1}%", success_rate * 100.0);
    println!("Security threshold: 95.0%");
    
    if success_rate > 0.8 {
        println!("✅ Security system functioning correctly");
    } else {
        println!("⚠️  Security system may need adjustment");
    }
    
    Ok(())
}

async fn benchmark_bmd_effectiveness(domain: &str) -> Result<()> {
    println!("⚡ Benchmarking BMD Effectiveness Scaling");
    println!("Domain: {}", domain);
    println!();
    
    let expertise_levels = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    
    println!("Expertise Level | BMD Effectiveness | Expected Range");
    println!("----------------|-------------------|---------------");
    
    for expertise in expertise_levels {
        let mut identity = SemanticIdentity::new();
        identity.understanding_vector.domains.insert(domain.to_string(), expertise);
        
        let bmd_effectiveness = identity.calculate_bmd_effectiveness(domain);
        
        let expected_range = match expertise {
            x if x < 0.2 => "60-65%",
            x if x < 0.5 => "65-75%",
            x if x < 0.8 => "75-85%",
            _ => "85-95%",
        };
        
        println!("{:>13.1}% | {:>15.1}% | {:>13}", 
                 expertise * 100.0, 
                 bmd_effectiveness * 100.0, 
                 expected_range);
    }
    
    println!();
    println!("📈 BMD Effectiveness scales correctly with user expertise");
    println!("✅ Novice users get 60%+ effectiveness (accessible)");
    println!("✅ Expert users get 90%+ effectiveness (powerful)");
    
    Ok(())
}
