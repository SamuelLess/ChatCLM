use std::ffi::{c_uint, c_void};
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::string::String;

use glob::glob;
use tiktoken_rs::p50k_base;
use zstd;
use zstd::dict::{DecoderDictionary, EncoderDictionary};
use zstd_sys;
use zstd_sys::{ZDICT_isError};
use zstd_sys::ZDICT_optimizeTrainFromBuffer_fastCover;

// https://wortschatz.uni-leipzig.de/en/download/English
const DATA_PATH: &str = "./data";
const DICT_SIZE_BYTES: usize = 1024 * 1024;

type Token = usize;

pub fn tokens_to_bytes(tokens: Vec<Token>) -> Vec<u8> {
    tokens.iter().flat_map(|x| (*x as u16).to_le_bytes()).collect()
}

pub fn read_dict() -> Vec<u8> {
    // read dictionary from file
    let mut file = File::open("model.zstd_dict").unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    buffer
}

pub fn decompress_to_tokens(compressed: &[u8]) -> Vec<Token> {
    let dict = DecoderDictionary::copy(&read_dict());
    let mut reader = zstd::stream::read::Decoder::with_prepared_dictionary(compressed, &dict).unwrap();

    let mut decompressed = Vec::new();
    reader.read_to_end(&mut decompressed).unwrap();

    let mut tokens = Vec::new();
    for i in decompressed.chunks(2) {
        tokens.push(u16::from_le_bytes([i[0], i[1]]) as Token);
    }

    tokens
}

pub fn compress(tokens: Vec<Token>) -> Vec<u8> {
    let raw_data = tokens_to_bytes(tokens);


    let dict = EncoderDictionary::copy(&read_dict(), 3);


    // memory writer
    let mut writer = zstd::stream::write::Encoder::with_prepared_dictionary(Vec::new(), &dict).unwrap();

    // write dictionary to memory writer
    writer.write_all(&raw_data).unwrap();

    // flush memory writer
    let compressed = writer.finish().unwrap();
    println!("Compressed size: {}", compressed.len());
    compressed
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

        let mut parameters = zstd_sys::ZDICT_fastCover_params_t {
            k: 0,
            d: 0,
            f: 20,
            steps: 4,
            nbThreads: 8,
            splitPoint: 0.0,
            accel: 1,
            shrinkDict: 0,
            shrinkDictMaxRegression: 0,
            zParams: zstd_sys::ZDICT_params_t {
                compressionLevel: 0,
                notificationLevel: 3,
                dictID: 0,
            },
        };
        let size = ZDICT_optimizeTrainFromBuffer_fastCover(
            buffer.as_mut_ptr() as *mut c_void,
            DICT_SIZE_BYTES,
            raw_data.as_ptr() as *mut c_void,
            sizes.as_ptr(),
            sizes.len() as c_uint,

            &mut parameters,
        );

        if ZDICT_isError(size) != 0 {
            panic!("Failed to train dictionary");
        }

        buffer.resize(size, 0);
    }

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
    let tokenizer = p50k_base().unwrap();

    reader
        .lines()
        .flat_map(|x| tokenizer.encode_ordinary(&x.expect("")))
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
    use zstd::bulk::decompress;

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

    #[test]
    fn compress_data() {
        let tokens = load_datasets();

        let start = &tokens[1000..2000];
        let compressed = compress(start.to_vec());
        println!("Compressed size: {}", compressed.len());
        println!("Decompressed size: {}", tokens_to_bytes(start.to_vec()).len());
        println!("Compressed size without dict: {}", zstd::bulk::compress(&tokens_to_bytes(start.to_vec()), 3).unwrap().len());
        let decompressed = decompress_to_tokens(&compressed);

        assert_eq!(decompressed, start);
    }
}