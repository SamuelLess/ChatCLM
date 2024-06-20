pub mod trainer;
pub mod training_options;
pub mod clm_model;
pub mod dataset;
pub mod evaluation;
pub mod ensemble_model;
mod tokenizer;

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
pub const MAX_TOKEN: Token = 50280; // Please update if u use another tokenizer!!!!

pub fn tokens_to_bytes(tokens: &Vec<Token>) -> Vec<u8> {
    tokens.iter().flat_map(|x| (*x).to_be_bytes()).collect()
}



const INFERENCE_COMPRESSION_LEVEL: i32 = 1;

pub mod tests {
    use rand::distributions::Uniform;
    use crate::backend::*;
    use crate::backend::dataset::Dataset;
    use crate::backend::trainer::train_model;
    use crate::backend::training_options::TrainingOptions;

    pub fn random_tokens(n: usize) -> Vec<Token> {
        rand::thread_rng().sample_iter(Uniform::from(0..MAX_TOKEN)).take(n).collect_vec()
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
        let start: Vec<Token> = clm.tokenizer.encode("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin tincidunt urna nisl, non molestie velit aliquam nec. In in erat id est porttitor efficitur ac eleifend ex. Nam auctor lacus urna, a sodales metus bibendum ut. Vestibulum vulputate facilisis ultrices. Vestibulum ut euismod erat. Maecenas pretium egestas nunc, non efficitur eros interdum eget. Suspendisse eleifend augue eu viverra rutrum. Phasellus non elementum erat, sit amet ultrices nunc. Sed facilisis at ipsum nec sagittis. Nulla non placerat purus. Pellentesque sed mollis enim. Praesent tincidunt purus id tellus tristique, ut rhoncus justo fringilla. Suspendisse fermentum ultrices dolor, vel mollis enim. Aliquam eros.");
        let compressed = clm.compress(&start);
        let decompressed = clm.decompress_to_tokens(&compressed);

        assert_eq!(decompressed, start);
    }

    #[test]
    fn save_and_load_model() {
        let data: Vec<Token> = random_tokens(100);
        let training_data = (0usize..10).map(|_| data.clone()).collect_vec();

        let model = train_model(&training_data, &TrainingOptions::new());
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
        let data: Vec<Token> = random_tokens(50);
        
        let training_data = Dataset::from_data((0usize..10).map(|_| data.clone()).collect_vec());

        let trained_model = train_model(training_data.get_data(), &TrainingOptions::new());
        let untrained_model = train_model(&Vec::new(), &TrainingOptions::default());

        let compressed = trained_model.compress(&data);
        let compressed_no_dict = untrained_model.compress(&data);

        println!("Compressed size with dict: {}, without dict: {}", compressed.len(), compressed_no_dict.len());

        assert!(compressed.len() < compressed_no_dict.len());
    }
}
