use monkey_tail_cli::cli::run_cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    run_cli().await
}