use std::fmt::{Debug, Display};
use glob::glob;

use tiktoken_rs::p50k_base;
use zstd;

const DATA_PATH: &str = "./data";

type Token = usize;

pub fn tokenize(str: &str) -> Vec<Token> {
    let tokenizer = p50k_base().unwrap();
    tokenizer.encode_with_special_tokens(str)
}

pub fn  compress(tokens: Vec<Token>) -> Vec<u8> {
    zstd::encode_all(tokens.as)
}


fn load_datasets() -> Vec<String> {
    find_datasets()
        .iter()
        .map(|x| std::fs::read_to_string(x).unwrap())
        .fold("", |acc, x| acc + &x)
        .collect()
}

fn find_datasets() -> Vec<String> {
    glob(&format!("{}/**/*-sentences.txt", DATA_PATH))
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)
        .map(|x| x.display().to_string())
        .collect()
}

mod tests {
    // https://wortschatz.uni-leipzig.de/en/download/English

    use crate::backend::find_datasets;

    #[test]
    fn test_find_dataset() {
        find_datasets().iter().for_each(|x| println!("Path found: {:?}", x));
    }

    #[test]
    fn test_load_datasets() {
        let datasets = crate::backend::load_datasets();
        datasets.iter().for_each(|x| println!("Dataset: {:?}", x));
    }
}