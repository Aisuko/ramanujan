use std::io::Read;
use std::process::ExitCode;

use ramanujan::data::{Dataset, convert_file};
use ramanujan::harness::Request;

const USAGE: &str = "usage:
  ramanujan prepare < request.json
  ramanujan convert --dataset qmsum|cnndm --src <raw.jsonl> --out <examples.jsonl> [--limit N]";

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn prepare() -> ramanujan::Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let request: Request = serde_json::from_str(&input)?;
    println!("{}", serde_json::to_string_pretty(&request.prepare()?)?);
    Ok(())
}

fn convert(args: &[String]) -> ramanujan::Result<()> {
    let need = |name: &str| {
        flag(args, name)
            .ok_or_else(|| ramanujan::Error::Invalid(format!("missing {name}\n{USAGE}")))
    };
    let dataset = Dataset::parse(&need("--dataset")?)?;
    let limit = match flag(args, "--limit") {
        Some(l) => Some(
            l.parse()
                .map_err(|_| ramanujan::Error::Invalid(format!("bad --limit {l}")))?,
        ),
        None => None,
    };
    let stats = convert_file(dataset, need("--src")?, need("--out")?, limit)?;
    eprintln!(
        "read {} written {} skipped {}",
        stats.read, stats.written, stats.skipped
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("prepare") => prepare(),
        Some("convert") => convert(&args[1..]),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
