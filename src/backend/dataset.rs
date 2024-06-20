use std::fs::File;
use std::io::BufWriter;
use std::io::{BufRead, BufReader};

use glob::glob;
use rand::seq::SliceRandom;
use rayon::iter::*;

use regex::Regex;
use tiktoken_rs::{CoreBPE, p50k_base};

use serde::{Deserialize, Serialize};
use rmp_serde::{Deserializer, Serializer};

use crate::backend::{DATA_PATH, Token};

#[derive(Serialize, Deserialize, Clone)]
pub struct Dataset {
    data: Vec<Vec<Token>>,
}



impl Dataset {
    fn locate_data_files() -> Vec<String> {
        glob(&format!("{}/**/*-sentences.txt", DATA_PATH))
            .expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .map(|x| x.display().to_string())
            .collect()
    }

    fn compute_from_files(files: Vec<String>) -> Dataset {
        let tokenizer = Self::get_tokenizer();
        let re = Regex::new(r"^\d+\s").unwrap();

        let pb = indicatif::ProgressBar::new(0);
        pb.set_style(indicatif::ProgressStyle::default_bar().template("{msg} {bar:60.cyan/blue} {pos}/{len} {per_sec}").unwrap());
        pb.set_message("Tokenizing");
        pb.set_length(files.iter().map(|file|
              File::open(file).unwrap().metadata().unwrap().len()
        ).sum());


        let tokens = files.iter()
            .flat_map(|filename| {
                let file = File::open(filename).unwrap();
                BufReader::new(file).lines()
            })
            .par_bridge()
            .map(|x| x.expect("Failed to read line"))
            .map(|x| {pb.inc(x.len() as u64); x})
            .map(|x| re.replace_all(&x, "").to_string())
            .map(|x| tokenizer.encode_ordinary(&x) as Vec<Token>)
            .collect();

        pb.finish();

        println!("Shuffling dataset");
        let mut dataset = Dataset { data: tokens };
        dataset.shuffle();
        dataset

    }

    pub fn split_train_test(&self, train_size: f32) -> (Dataset, Dataset) {
        let train_size = (self.data.len() as f32 * train_size).round() as usize;
        let (train, test) = self.data.split_at(train_size);
        (Dataset { data: train.to_vec() }, Dataset { data: test.to_vec() })
    }

    fn save_to_file(&self, filename: &str) {
        let file = File::create(filename).unwrap();
        let mut writer = BufWriter::new(file);
        let mut serializer = Serializer::new(&mut writer);
        self.serialize(&mut serializer).unwrap();
    }

    pub fn load_from_file(filename: &str) -> Dataset {
        let file = File::open(filename).unwrap();
        let mut reader = BufReader::new(file);
        let mut deserializer = Deserializer::new(&mut reader);
        Dataset::deserialize(&mut deserializer).unwrap()
    }

    pub fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        self.data.shuffle(&mut rng);
    }

    pub fn load_or_compute(filename: &str) -> Dataset {
        if std::path::Path::new(filename).exists() {
            Dataset::load_from_file(filename)
        } else {
            let dataset = Dataset::compute_from_files(Dataset::locate_data_files());
            dataset.save_to_file(filename);
            dataset
        }
    }

    pub fn get_data(&self) -> &Vec<Vec<Token>> {
        &self.data
    }

    pub fn get_data_out(self) -> Vec<Vec<Token>> {
        self.data
    }

    pub fn from_data(data: Vec<Vec<Token>>) -> Dataset {
        Dataset { data }
    }

    pub fn shrink_to_size(&self, tokens: usize) -> Dataset {
        let mut total_tokens  = 0;
        let mut new_dataset = Vec::new();

        for line in self.data.iter() {
            if total_tokens + line.len() <= tokens {
                new_dataset.push(line.clone());
                total_tokens += line.len();
            } else if total_tokens > tokens {
                continue;
            } else if total_tokens + line.len() > tokens {
                let partial_vec = line[0..(tokens-total_tokens)].to_vec();
                total_tokens += partial_vec.len();
                new_dataset.push(partial_vec);
            }
        }

        Self::from_data(new_dataset)
    }

    // Join adjacent lines until they reach a certain size
    pub fn join_lines(self, chunk_size: usize) -> Dataset {
        let new_data = self.data.into_iter().fold(Vec::new(), |mut acc, line| {
            if acc.is_empty() {
                acc.push(line);
            } else if acc.last().unwrap().len() < chunk_size {
                acc.last_mut().unwrap().extend(line);
            } else {
                acc.push(line);
            }
            acc
        });
        
        Self::from_data(new_data)
    }

    pub fn split_into_chunks(&self, chunk_count: usize) -> impl Iterator<Item=Dataset>  + '_{

        if self.data.is_empty() {
           panic!("Cannot split empty dataset into chunks");
        }

        let chunk_size = (self.data.len() as f64 / chunk_count as f64).ceil() as usize;
        // split into n chunks, each containing chunk_size sentences
        self.data.chunks(chunk_size).map(|x| Dataset::from_data(x.to_vec()))
    }
    pub fn empty() -> Dataset {
        Dataset { data: Vec::new() }
    }

    fn get_tokenizer() -> CoreBPE {
        p50k_base().unwrap()
    }
    pub fn tokenize(text: &str) -> Vec<Token> {
        Self::get_tokenizer().encode_ordinary(text) as Vec<Token>
    }
    
    pub fn detokenize(tokens: Vec<Token>) -> String {
        Self::get_tokenizer().decode(tokens).unwrap_or("<error>".to_string())
    }

    #[cfg(test)]
    pub fn test_dataset() -> Dataset {
        let mut data = Vec::new();
        for _ in 0..20 {
            let mut sentence = Vec::new();
            for _ in 0..100 {
                sentence.push(rand::random());
            }
            data.push(sentence);
        }
        Dataset { data }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use crate::backend::dataset::Dataset;

    #[test]
    fn test_locate_data_files() {
        let files = Dataset::locate_data_files();
        assert_ne!(files.len(), 0);
    }

    #[test]
    fn save_and_load_dataset() {
        let start_time = std::time::Instant::now();
        let dataset = Dataset::compute_from_files(vec!["./data/tests.txt".to_string()]);
        println!("Dataset computed in {:?}", start_time.elapsed());
        dataset.save_to_file("dataset.msgpack");

        let load_time = std::time::Instant::now();
        let loaded_dataset = Dataset::load_from_file("dataset.msgpack");
        println!("Dataset loaded in {:?}", load_time.elapsed());
        println!("Computed file size: {:?}", std::fs::metadata("dataset.msgpack").unwrap().len());
        println!("Original file size: {:?}", std::fs::metadata("./data/tests.txt").unwrap().len());

        assert_eq!(dataset.data.len(), loaded_dataset.data.len());

        // deep comparison
        for (a, b) in dataset.data.iter().zip(loaded_dataset.data.iter()) {
            assert_eq!(a, b);
        }

        // cleanup
        std::fs::remove_file("dataset.msgpack").unwrap();
    }


    #[test]
    fn test_compute_from_files() {
        let dataset = Dataset::compute_from_files(vec!["data/tests.txt".to_string()]);
        assert_eq!(dataset.data.len(), 1000);

        // no sentence should be empty
        assert_eq!(dataset.data.iter().filter(|x| x.is_empty()).count(), 0);
    }

    #[test]
    fn shrink_dataset() {
        let dataset = Dataset::compute_from_files(vec!["data/tests.txt".to_string()]);
        let shrunk = dataset.shrink_to_size(1000);
        assert_eq!(shrunk.data.iter().map(|x| x.len()).sum::<usize>(), 1000);
    }


    #[test]
    fn split_into_chunks() {
        let dataset = Dataset::compute_from_files(vec!["data/tests.txt".to_string()]);
        let chunks = dataset.split_into_chunks(10).collect_vec();
        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks.iter().map(|x| x.data.len()).sum::<usize>(), dataset.data.len());
    }
    
    #[test]
    fn join_lines() {
        let dataset = Dataset::compute_from_files(vec!["data/tests.txt".to_string()]);
        let joined = dataset.join_lines(100);
        // the last line can be shorter than 100, it doesn't matter
        let min_len = joined.data.iter().rev().skip(1).map(|x| x.len()).min().unwrap();
        println!("Min len: {}", min_len);
        assert!(min_len >= 100);
    }
}