use std::fmt::{Debug, Display};
use glob::glob;

use tiktoken_rs::p50k_base;

const DATA_PATH: &str = "./data";

pub fn tokenize(str: &str) -> String {
    let tokens 
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
}