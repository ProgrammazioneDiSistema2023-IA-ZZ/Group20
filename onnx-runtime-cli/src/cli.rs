use std::{path::PathBuf, str::FromStr};

use clap::Parser;

use strum::Display;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Input image path.
    /// Example: --input /path/to/image1.jpg --input /path/to/image2.jpg
    #[arg(short, long)]
    pub input: Vec<PathBuf>,

    /// Model name.
    /// Supported models: resnet18, mobilenet
    /// Default: mobilenet
    #[arg(short, long, default_value = "mobilenet")]
    pub model: Model,

    /// Number of threads to use for inference to parallelize a single operation. Must be greater than 0 and less than 65536.
    /// Example: --threads 8
    #[arg(short, long, default_value = "4", value_parser = clap::value_parser!(u16).range(1..))]
    pub threads: u16,

    /// Number of top infered class probabilities to show. Must be greater than 0 and less or equal than 1000.
    /// Example: --show 10
    #[arg(short, long, default_value = "5", value_parser = clap::value_parser!(u16).range(1..=1000))]
    pub show: u16,

    /// Input parameters in the format name:value.
    /// Example: --params N:1 --params BATCH_SIZE:1
    #[arg(short, long)]
    pub params: Vec<InputParameter>,
}

#[derive(Clone, Debug)]
pub struct InputParameter {
    pub name: String,
    pub value: usize,
}

impl FromStr for InputParameter {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split(':');
        let name = parts
            .next()
            .ok_or_else(|| "Invalid input parameter".to_string())?;
        let value = parts
            .next()
            .ok_or_else(|| "Invalid input parameter".to_string())?
            .parse()
            .map_err(|err| format!("Invalid input parameter value: {}", err))?;
        Ok(Self {
            name: name.to_string(),
            value,
        })
    }
}

#[derive(Debug, Clone, Display)]
pub enum Model {
    #[strum(serialize = "resnet18")]
    Resnet18,
    #[strum(serialize = "mobilenet")]
    Mobilenet,
}

impl FromStr for Model {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "resnet18" => Ok(Self::Resnet18),
            "mobilenet" => Ok(Self::Mobilenet),
            _ if s.starts_with('r') => Ok(Self::Resnet18),
            _ if s.starts_with('m') => Ok(Self::Mobilenet),
            _ => Err(format!("Invalid model name: {}", s)),
        }
    }
}
