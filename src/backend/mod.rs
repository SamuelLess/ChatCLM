mod trainer;
mod training_options;
pub mod clm_model;

use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::string::String;

use glob::glob;
use itertools::Itertools;
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::iter::*;
use regex::Regex;
use tiktoken_rs::p50k_base;
use zstd;
use zstd::dict::DecoderDictionary;
use zstd_sys;
use clm_model::ClmModel;

// https://wortschatz.uni-leipzig.de/en/download/English
const DATA_PATH: &str = "./data";

pub type Token = usize;
const BYTES_PER_TOKEN: usize = std::mem::size_of::<Token>();
const MAX_TOKEN: Token = 50280; // Please update if u use another tokenizer!!!!

pub fn tokens_to_bytes(tokens: &Vec<Token>) -> Vec<u8> {
    tokens.iter().flat_map(|x| (*x).to_be_bytes()).collect()
}

fn find_datasets() -> Vec<String> {
    glob(&format!("{}/**/*-sentences.txt", DATA_PATH))
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)
        .map(|x| x.display().to_string())
        .collect()
}

pub fn tokenize_files() -> (Vec<Token>, Vec<usize>) {
    let tokenizer = p50k_base().unwrap();
    let re = Regex::new(r"^\d+\s").unwrap();

    let pb = indicatif::ProgressBar::new(0);
    pb.set_style(indicatif::ProgressStyle::default_bar().template("{msg} {bar:60.cyan/blue} {pos}/{len} {per_sec}").unwrap());
    pb.set_message("Tokenizing");
    pb.set_length(find_datasets().iter().map(|file|
        BufReader::new(File::open(file).unwrap())
            .lines()
            .count() as u64
    ).sum());


    find_datasets()
        .iter()
        .flat_map(|filename| {
            let file = File::open(filename).unwrap();
            println!("Reading file: {:?}", filename);
            BufReader::new(file).lines()
        })
        .par_bridge()
        .map(|x| x.expect("Failed to read line"))
        .map(|x| re.replace_all(&x, "").to_string())
        .map(|x| tokenizer.encode_ordinary(&x) as Vec<Token>)
        .map(|x| {
            let len = x.len();
            pb.inc(1);
            (x, vec![len])
        })
        .reduce(
            || (Vec::new(), Vec::new()),
            |(mut tok_acc, mut len_acc), (tok, len)| {
                tok_acc.extend_from_slice(&tok);
                len_acc.extend_from_slice(&len);
                (tok_acc, len_acc)
            },
        )
}



const INFERENCE_COMPRESSION_LEVEL: i32 = 1;

pub mod tests {
    use rand::distributions::Uniform;
    use crate::backend::*;
    use crate::backend::trainer::train_model;
    use crate::backend::training_options::TrainingOptions;
    
    fn random_tokens(n: usize) -> Vec<Token> {
        rand::thread_rng().sample_iter(Uniform::from(0..MAX_TOKEN)).take(n).collect_vec()
    }
    
    #[test]
    #[ignore]
    fn test_find_dataset() {
        find_datasets()
            .iter()
            .for_each(|x| println!("Path found: {:?}", x));
    }

    #[test]
    #[ignore]
    fn test_load_datasets() {
        let (tokens, _sizes) = tokenize_files();
        let max = tokens.iter().max().unwrap();

        println!("Dataset: {:?}, max: {:?}", tokens.len(), max);
    }

    pub fn predict_loop() {
        // create text for a test prompt
        //let mut prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy".to_string();

        let mut prompt = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin tincidunt urna nisl, non molestie velit aliquam nec. In in erat id est porttitor efficitur ac eleifend ex. Nam auctor lacus urna, a sodales metus bibendum ut. Vestibulum vulputate facilisis ultrices. Vestibulum ut euismod erat. Maecenas pretium egestas nunc, non efficitur eros interdum eget. Suspendisse eleifend augue eu viverra rutrum. Phasellus non elementum erat, sit amet ultrices nunc. Sed facilisis at ipsum nec sagittis. Nulla non placerat purus. Pellentesque sed mollis enim. Praesent tincidunt purus id tellus tristique, ut ".to_string();

        let clm = ClmModel::from_checkpoint("model.zstd_dict");

        println!("{}", prompt);

        for _ in 0..100 {
            prompt = clm.predict_next(prompt.clone(), 0, 0);
            println!("{}", prompt)
        }
    }

    #[test]
    fn can_compress_and_decompress() {
        let clm = ClmModel::from_buffer(Vec::new());
        let start: Vec<Token> = clm.tokenizer.encode_ordinary("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin tincidunt urna nisl, non molestie velit aliquam nec. In in erat id est porttitor efficitur ac eleifend ex. Nam auctor lacus urna, a sodales metus bibendum ut. Vestibulum vulputate facilisis ultrices. Vestibulum ut euismod erat. Maecenas pretium egestas nunc, non efficitur eros interdum eget. Suspendisse eleifend augue eu viverra rutrum. Phasellus non elementum erat, sit amet ultrices nunc. Sed facilisis at ipsum nec sagittis. Nulla non placerat purus. Pellentesque sed mollis enim. Praesent tincidunt purus id tellus tristique, ut rhoncus justo fringilla. Suspendisse fermentum ultrices dolor, vel mollis enim. Aliquam eros.");
        let compressed = clm.compress(&start);
        let decompressed = clm.decompress_to_tokens(&compressed);

        assert_eq!(decompressed, start);
    }

    #[test]
    fn save_and_load_model() {
        let data: Vec<Token> = random_tokens(100);
        let training_data = (0usize..10).map(|_| data.clone()).collect_vec();
        
        let model = train_model(training_data, TrainingOptions::new());
        model.save_checkpoint("model_test.zstd_dict");
        let loaded_model = ClmModel::from_checkpoint("model_test.zstd_dict");
        
        let compressed = model.compress(&data);
        let compressed_loaded = loaded_model.compress(&data);
        
        assert_eq!(compressed, compressed_loaded);
        
        // cleanup
        std::fs::remove_file("model_test.zstd_dict").unwrap();
    }
    #[test]
    fn dictionary_helps_compression() {
        let data: Vec<Token> = random_tokens(100);
        
        let training_data = (0usize..10).map(|_| data.clone()).collect_vec();

        let trained_model = train_model(training_data, TrainingOptions::new());
        let untrained_model = train_model(Vec::new(), TrainingOptions::default());

        let compressed = trained_model.compress(&data);
        let compressed_no_dict = untrained_model.compress(&data);

        println!("Compressed size with dict: {}, without dict: {}", compressed.len(), compressed_no_dict.len());

        assert!(compressed.len() < compressed_no_dict.len());
    }
}
