use std::ffi::{c_uint, c_void};
use std::fmt::{Debug, Display};
use std::string::String;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use glob::glob;
use zstd::dict::{DDict, from_continuous};

use zstd_sys;
use tiktoken_rs::p50k_base;
use zstd;
use zstd_sys::ZDICT_trainFromBuffer;

// https://wortschatz.uni-leipzig.de/en/download/English
const DATA_PATH: &str = "./data";
const DICT_SIZE_BYTES : usize = 1024 * 1024 * 1024;

type Token = usize;

pub fn tokenize(str: &str) -> Vec<Token> {
    let tokenizer = p50k_base().unwrap();
    tokenizer.encode_with_special_tokens(str)
}

pub fn tokens_to_bytes(tokens: Vec<Token>) -> Vec<u8> {
    tokens.iter().map(|x| x.to_le_bytes()).flatten().collect()
}

pub fn compress(tokens: Vec<Token>) {
    let raw_data = tokens_to_bytes(tokens);
    //zstd::encode_all(&raw_data, 0).unwrap()
}


pub fn create_dictionary() {
    let tokens = load_datasets();
    let raw_data =  tokens_to_bytes(tokens);


    let mut buffer = vec![0u8; DICT_SIZE_BYTES];

    let sample_size = 1000;
    let sizes_count = (raw_data.len() / sample_size) + 1;
    let mut sizes = vec![sample_size; sizes_count];
    sizes[sizes_count - 1] = raw_data.len() % sample_size;

    unsafe {
        ZDICT_trainFromBuffer(
            buffer.as_mut_ptr() as *mut c_void,
            DICT_SIZE_BYTES,
            raw_data.as_ptr() as *mut c_void,
            sizes.as_ptr(),
            sizes.len() as c_uint
        );
    }


    let dict = DDict::create(&buffer);
    
    // write buffer to file 
    let mut file = File::create("model.zstd_dict").unwrap();
    file.write_all(&buffer).unwrap();
}

fn load_datasets() -> Vec<Token> {
    find_datasets()
        .iter()
        .flat_map(|filename| tokenize_file(filename))
        .collect()
}

fn tokenize_file(filename: &str) -> Vec<Token> {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    reader
        .lines()
        .flat_map(|x| tokenize(&x.expect("")))
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
    use crate::backend::*;

    #[test]
    fn test_find_dataset() {
        find_datasets()
            .iter()
            .for_each(|x| println!("Path found: {:?}", x));
    }

    #[test]
    fn test_load_datasets() {
        println!("Dataset: {:?}", load_datasets());
    }
    
    #[test]
    fn train_dict() {
        create_dictionary();
    }
}