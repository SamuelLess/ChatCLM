use std::ffi::{c_uint, c_void};
use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::string::String;

use glob::glob;
use itertools::Itertools;
use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;
use rayon::iter::*;
use regex::Regex;
use tiktoken_rs::{CoreBPE, p50k_base};
use zstd;
use zstd::dict::{DecoderDictionary, EncoderDictionary};
use zstd_sys;
use zstd_sys::ZDICT_isError;
use zstd_sys::ZDICT_optimizeTrainFromBuffer_fastCover;

// https://wortschatz.uni-leipzig.de/en/download/English
const DATA_PATH: &str = "./data";

const MAX_TOKEN: u16 = 50280; // Please update if u use another tokenizer!!!!


pub struct CLM<'a> {
    dict: EncoderDictionary<'a>,
    tokenizer: CoreBPE,
}

type Token = usize;

pub fn tokens_to_bytes(tokens: Vec<Token>) -> Vec<u8> {
    tokens.iter().flat_map(|x| (*x).to_be_bytes()).collect()
}

fn find_datasets() -> Vec<String> {
    glob(&format!("{}/**/*-sentences.txt", DATA_PATH))
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)
        .map(|x| x.display().to_string())
        .collect()
}


pub fn create_dictionary() {
    let (tokens, mut sizes) = tokenize_files();
    let raw_data = tokens_to_bytes(tokens);
    // multiply everything by 2 as tokens get to be compressed to u64
    sizes = sizes.iter().map(|x| x * 8).collect();

    let buffer_size = raw_data.len(); // 100% of training size

    assert_eq!(sizes.iter().sum::<usize>(), raw_data.len(), "Sizes sum doesn't match raw data size");

    let mut buffer = vec![0u8; buffer_size];

    let mut parameters = zstd_sys::ZDICT_fastCover_params_t {
        k: 50,
        d: 8,
        f: 25,
        steps: 4,
        nbThreads: 8,
        splitPoint: 0.0,
        accel: 1,
        shrinkDict: 0,
        shrinkDictMaxRegression: 0,
        zParams: zstd_sys::ZDICT_params_t {
            compressionLevel: 3,
            notificationLevel: 4,
            dictID: 0,
        },
    };

    let mut size = 0;
    unsafe {
        size = ZDICT_optimizeTrainFromBuffer_fastCover(
            buffer.as_mut_ptr() as *mut c_void,
            buffer_size,
            raw_data.as_ptr() as *mut c_void,
            sizes.as_ptr(),
            sizes.len() as c_uint,
            &mut parameters,
        );
        println!("Selected parameters: {:?}", parameters);
        println!("Dictionary size: {:?}", size);

        if ZDICT_isError(size) != 0 {
            panic!("Failed to train dictionary");
        }
    }

    println!("Dictionary trained, resizing buffer to size: {}", size);
    buffer.resize(size, 0);
    println!("Buffer resized {}", buffer.len());

    println!("Writing dictionary to file");
    // write buffer to file
    let mut file = File::create("./model.zstd_dict").unwrap();
    file.write_all(&buffer).unwrap();
    file.flush();
    file.sync_all();
    drop(file);
    println!("Dictionary written to file");
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

pub fn decompress_to_tokens(compressed: &[u8]) -> Vec<Token> {
    let dict = DecoderDictionary::copy(CLM::read_dict().as_slice());
    let mut reader = zstd::stream::read::Decoder::with_prepared_dictionary(compressed, &dict).unwrap();

    let mut decompressed = Vec::new();
    reader.read_to_end(&mut decompressed).unwrap();

    let mut tokens = Vec::new();
    for i in decompressed.chunks(2) {
        tokens.push(u16::from_le_bytes([i[0], i[1]]) as Token);
    }

    tokens
}

const INFERENCE_COMPRESSION_LEVEL: i32 = 1;

impl<'a> CLM<'a> {
    pub fn new() -> Self {
        let dict = EncoderDictionary::copy(CLM::read_dict().as_slice(), INFERENCE_COMPRESSION_LEVEL);
        let tokenizer = p50k_base().unwrap();
        Self { dict, tokenizer }
    }

    pub fn read_dict() -> Vec<u8> {
        // read dictionary from file
        let mut file = File::open("model.zstd_dict").unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        buffer
    }


    pub fn compress(&self, tokens: Vec<Token>) -> Vec<u8> {
        let raw_data = tokens_to_bytes(tokens);

        // memory writer
        let mut writer = zstd::stream::write::Encoder::with_prepared_dictionary(Vec::new(), &self.dict).unwrap();

        // write dictionary to memory writer
        writer.write_all(&raw_data).unwrap();

        let compressed = writer.finish().unwrap();
        compressed
    }

    pub fn predict_next(&self, prompt: String, depth: usize, width: usize) -> String {
        let mut tokens = self.tokenizer.encode_ordinary(&prompt);
        let (next_token, _size) = self.predict_tokens(tokens.clone(), depth, width);
        tokens.push(next_token);
        self.tokenizer.decode(tokens).unwrap_or(prompt + "<error>")
    }

    pub fn predict_diffusion(&self, prompt: String) -> String {
        let mut tokens = self.tokenizer.encode_ordinary(&prompt);
        let next_tokens = self.predict_tokens_diffusion(tokens.clone());
        tokens.extend_from_slice(&next_tokens);
        self.tokenizer.decode(tokens).unwrap_or(prompt)
    }

    fn predict_tokens(&self, tokens: Vec<Token>, depth: usize, width: usize) -> (Token, usize) {
        let sizes = (1..MAX_TOKEN)
            .par_bridge()
            .map(|x| x as Token)
            // .filter(|token| !tokens.as_slice()[tokens.len()-5..].contains(token))
            .map(|x| {
                let mut prompt = tokens.clone();
                prompt.push(x);
                (x, self.compress(prompt).len())
            });

        let mut c: Vec<(Token, usize)> = sizes.collect();

        c.shuffle(&mut thread_rng());

        c.sort_by(|a, b| a.1.cmp(&b.1));

        let mut hist = c.iter().map(|x| x.1).counts().iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
        hist.sort_by(|a, b| a.1.cmp(&b.1));


        println!("Predicting tokens: {:?}", hist);

        if depth == 0 {
            return c[0];
        }

        let mut best = (0, std::usize::MAX);
        for (token, _) in c.iter().take(width) {
            let (_next_token, next_compression) = self.predict_tokens(tokens.clone(), depth - 1, width / 2);
            if next_compression < best.1 {
                best = (*token, next_compression);
            }
        }

        best
    }

    fn compress_with_prompt(&self, prompt: &Vec<Token>, tokens: &Vec<Token>) -> usize {
        let mut prompt = prompt.clone();
        prompt.extend_from_slice(&tokens);
        self.compress(prompt).len()
    }

    fn predict_tokens_diffusion(&self, prompt: Vec<Token>) -> Vec<Token> {
        let mut rng = thread_rng();

        let mut current_tokens = (1..20).map(|_| thread_rng().gen_range(0..MAX_TOKEN) as Token).collect_vec();
        let mut current_compression = self.compress_with_prompt(&prompt, &current_tokens);

        for i in 0..1000000 {
            let mut next_tokens = current_tokens.clone();
            let random_index = rng.gen_range(0..next_tokens.len());
            next_tokens[random_index] = rng.gen_range(0..MAX_TOKEN) as Token;

            let next_compression = self.compress_with_prompt(&prompt, &next_tokens);

            if next_compression < current_compression {
                current_tokens = next_tokens;
                current_compression = next_compression;

                println!("Iteration: {}, Compression: {}, Values: '{}'", i, current_compression, self.tokenizer.decode(current_tokens.clone()).unwrap_or("error".to_string()));
            }
        }

        current_tokens
    }
}

pub mod tests {
    use crate::backend::*;

    #[test]
    fn test_find_dataset() {
        find_datasets()
            .iter()
            .for_each(|x| println!("Path found: {:?}", x));
    }

    #[test]
    fn test_load_datasets() {
        let (tokens, _sizes) = tokenize_files();
        let max = tokens.iter().max().unwrap();

        println!("Dataset: {:?}, max: {:?}", tokens.len(), max);
    }

    #[test]
    fn train_dict() {
        create_dictionary();
    }

    pub fn predict_loop() {
        // create text for a test prompt
        //let mut prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy".to_string();

        let mut prompt = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin tincidunt urna nisl, non molestie velit aliquam nec. In in erat id est porttitor efficitur ac eleifend ex. Nam auctor lacus urna, a sodales metus bibendum ut. Vestibulum vulputate facilisis ultrices. Vestibulum ut euismod erat. Maecenas pretium egestas nunc, non efficitur eros interdum eget. Suspendisse eleifend augue eu viverra rutrum. Phasellus non elementum erat, sit amet ultrices nunc. Sed facilisis at ipsum nec sagittis. Nulla non placerat purus. Pellentesque sed mollis enim. Praesent tincidunt purus id tellus tristique, ut ".to_string();

        let clm = CLM::new();

        println!("{}", prompt);

        for _ in 0..100 {
            prompt = clm.predict_next(prompt.clone(), 0, 0);
            println!("{}", prompt)
        }
    }

    pub fn test_predict_token() {}

    #[test]
    fn compress_data() {
        let clm = CLM::new();
        let start: Vec<Token> = clm.tokenizer.encode_ordinary("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin tincidunt urna nisl, non molestie velit aliquam nec. In in erat id est porttitor efficitur ac eleifend ex. Nam auctor lacus urna, a sodales metus bibendum ut. Vestibulum vulputate facilisis ultrices. Vestibulum ut euismod erat. Maecenas pretium egestas nunc, non efficitur eros interdum eget. Suspendisse eleifend augue eu viverra rutrum. Phasellus non elementum erat, sit amet ultrices nunc. Sed facilisis at ipsum nec sagittis. Nulla non placerat purus. Pellentesque sed mollis enim. Praesent tincidunt purus id tellus tristique, ut rhoncus justo fringilla. Suspendisse fermentum ultrices dolor, vel mollis enim. Aliquam eros.");
        let compressed = clm.compress(start.clone());

        println!("Compressed size: {}", compressed.len());
        println!("Tokens size: {}", tokens_to_bytes(start.clone()).len());
        println!("Compressed size without dict: {}", zstd::bulk::compress(&tokens_to_bytes(start.clone().to_vec()), 3).unwrap().len());
        let decompressed = decompress_to_tokens(&compressed);

        assert_eq!(decompressed, start);
    }
}