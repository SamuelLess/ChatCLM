use std::collections::HashMap;
use zstd::dict::{DecoderDictionary, EncoderDictionary};
use tiktoken_rs::{CoreBPE, p50k_base};
use std::fs::File;
use rand::{thread_rng};
use std::io::{Read, Write};
use std::ops::Range;
use rayon::iter::{IterBridge, Map, ParallelBridge, ParallelIterator};
use rand::prelude::SliceRandom;
use itertools::Itertools;
use crate::backend;
use crate::backend::{INFERENCE_COMPRESSION_LEVEL, MAX_TOKEN, Token};

pub struct ClmModel<'a> {
    dict: EncoderDictionary<'a>,
    model_buffer: Vec<u8>,
    pub(crate) tokenizer: CoreBPE,
}

impl<'a> ClmModel<'a> {
    pub fn from_buffer(model_buffer: Vec<u8>) -> Self {
        let dict = EncoderDictionary::copy(&*model_buffer, INFERENCE_COMPRESSION_LEVEL);
        let tokenizer = p50k_base().unwrap();
        Self { dict, model_buffer, tokenizer }
    }

    pub fn from_checkpoint(path: &str) -> Self {
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        ClmModel::from_buffer(buffer)
    }
    
    pub fn save_checkpoint(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        file.write_all(&self.model_buffer).unwrap();
    }

    pub fn compress(&self, tokens: &Vec<Token>) -> Vec<u8> {
        let raw_data = backend::tokens_to_bytes(tokens);

        // Actual compression
        let mut writer = zstd::stream::write::Encoder::with_prepared_dictionary(Vec::new(), &self.dict).unwrap();
        writer.write_all(&raw_data).unwrap();
        let compressed = writer.finish().unwrap();
        compressed
    }

    pub fn predict_next(&self, prompt: String, depth: usize, width: usize) -> String {
        let mut tokens = self.tokenizer.encode_ordinary(&prompt);
        let (next_token, _size) = self.predict_tokens(&tokens, depth, width);
        tokens.push(next_token);
        self.tokenizer.decode(tokens).unwrap_or(prompt + "<error>")
    }

    fn get_next_token_sizes(&self, tokens: &Vec<Token>) -> Vec<(Token, usize)>{
         (1..MAX_TOKEN)
            .par_bridge()
            .map(|x| x as Token)
            .map(|x| {
                let mut prompt = tokens.clone();
                prompt.push(x);
                (x, self.compress(&prompt).len())
            }).collect()
    }
    pub(crate) fn next_token_distribution(&self, tokens: &Vec<Token>) -> HashMap<Token, f64> {
        let c = self.get_next_token_sizes(tokens);
        let sum = c.iter().map(|x| x.1).sum::<usize>() as f64;
        c.iter().map(|(k, v)| (*k, *v as f64 / sum)).collect()
    }

    fn predict_tokens(&self, tokens: &Vec<Token>, depth: usize, width: usize) -> (Token, usize) {
        let mut c = self.get_next_token_sizes(tokens);

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
            let (_next_token, next_compression) = self.predict_tokens(tokens, depth - 1, width / 2);
            if next_compression < best.1 {
                best = (*token, next_compression);
            }
        }
        best
    }

    pub(crate) fn compress_together(&self, prompt: &Vec<Token>, tokens: &Vec<Token>) -> usize {
        let mut prompt = prompt.clone();
        prompt.extend_from_slice(&tokens);
        self.compress(&prompt).len()
    }

    pub fn decompress_to_tokens(&self, compressed: &[u8]) -> Vec<Token> {
        let dict = DecoderDictionary::copy(self.model_buffer.as_slice());
        let mut reader = zstd::stream::read::Decoder::with_prepared_dictionary(compressed, &dict).unwrap();

        let mut decompressed = Vec::new();
        reader.read_to_end(&mut decompressed).unwrap();

        let mut tokens = Vec::new();
        for i in decompressed.chunks(backend::BYTES_PER_TOKEN) {
            tokens.push(Token::from_be_bytes([i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]]));
        }

        tokens
    }

    pub fn evaluate(&self, test_data: &Vec<Vec<Token>>) -> f64 {
        let mut total = 0;
        let mut correct = 0;
        for tokens in test_data {
            let compressed = self.compress(tokens);
            let decompressed = self.decompress_to_tokens(&compressed);
            total += tokens.len();
            correct += tokens.iter().zip(decompressed.iter()).filter(|(a, b)| a == b).count();
        }
        correct as f64 / total as f64
    }
}
