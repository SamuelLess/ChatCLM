use std::ffi::{c_uint, c_void};
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::string::String;

use glob::glob;
use rayon::iter::*;
use tiktoken_rs::{CoreBPE, p50k_base};
use zstd;
use zstd::dict::{DecoderDictionary, EncoderDictionary};
use zstd_sys;
use zstd_sys::ZDICT_isError;
use zstd_sys::ZDICT_optimizeTrainFromBuffer_fastCover;

// https://wortschatz.uni-leipzig.de/en/download/English
const DATA_PATH: &str = "./data";
const DICT_SIZE_BYTES: usize = 1024 * 1024;

const MAX_TOKEN: u16 = 50280; // Please update if u use another tokenizer!!!!


pub struct CLM<'a> {
    dict: EncoderDictionary<'a>,
    tokenizer: CoreBPE,
}

type Token = usize;

pub fn tokens_to_bytes(tokens: Vec<Token>) -> Vec<u8> {
    tokens.iter().flat_map(|x| (*x as u16).to_le_bytes()).collect()
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

    // multiply everything by 2 as tokens get to be compressed to u16
    sizes = sizes.iter().map(|x| x * 2).collect();

    let dict_size = raw_data.len() / 10; // 10% of training size
    let mut buffer = vec![0u8; dict_size];

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
            dict_size,
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

pub fn tokenize_files() -> (Vec<Token>, Vec<usize>) {
    let tokenizer = p50k_base().unwrap();

    find_datasets()
        .iter()
        .flat_map(|filename| {
            let file = File::open(filename).unwrap();
            BufReader::new(file).lines()
        })
        .par_bridge()
        .map(|x| x.expect("Failed to read line"))
        .map(|x| x.trim_start_matches(""))
        .map(|x| (tokenizer.encode_ordinary(&x) as Vec<Token>, vec![x.len()]))
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

impl<'a> CLM <'a> {
    pub fn new() -> Self {
        let dict = EncoderDictionary::copy(CLM::read_dict().as_slice(), 3);
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

    pub fn predict_next(&self, prompt: String) -> String {
        let mut tokens = self.tokenizer.encode_ordinary(&prompt);
        let next_token = self.predict_token(tokens.clone());
        tokens.push(next_token);
        self.tokenizer.decode(tokens).unwrap()
    }

    fn predict_token(&self, tokens: Vec<Token>) -> Token {
        (1..MAX_TOKEN)
            .par_bridge()
            .map(|x| x as Token)
            .map(|x| {
                let mut prompt = tokens.clone();
                prompt.push(x);
                (x, self.compress(prompt).len())
            })
            .reduce(|| (0u16 as Token, usize::MAX), |(min_tok, min_size), (tok, size)| {
                if size < min_size { (tok, size) } else { (min_tok, min_size) }
            })
            .0
    }

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
        let (tokens, sizes) = tokenize_files();
        let max = tokens.iter().max().unwrap();

        println!("Dataset: {:?}, max: {:?}", tokens.len(), max);
    }

    #[test]
    fn train_dict() {
        create_dictionary();
    }

    #[test]
    fn test_predict_token() {
        let mut prompt = "A frontman, often seen as the face and voice of a musical band, plays a pivotal role in defining the band's identity and engaging with the audience. This role extends beyond merely singing; it encompasses a dynamic blend of charisma, stage presence, and the ability to connect with the crowd. The frontman leads the performance, sets the tone, and often shapes the band's public image through their interactions both on and off the stage.
Historically, iconic frontmen like Freddie Mercury of Queen, Mick Jagger of The Rolling Stones, and Robert Plant of Led Zeppelin have left indelible marks on the music industry with their electrifying performances and distinctive personas. Their ability to captivate audiences not only relied on vocal prowess but also on their unique style".to_string();
        let clm = CLM::new();

        println!("{}", prompt);

        for _ in 0..10 {
            prompt = clm.predict_next(prompt.clone());
            println!("{}", prompt)
        }

    }

    //#[test]
    //fn compress_data() {
    //    let (tokens, sizes) = tokenize_files();
    //
    //    let start = &tokens[1000..2000];
    //    let dict = EncoderDictionary::copy(&read_dict(), 3);
    //    let compressed = compress(start.to_vec(), &dict);
    //
    //    println!("Compressed size: {}", compressed.len());
    //    println!("Decompressed size: {}", tokens_to_bytes(start.to_vec()).len());
    //    println!("Compressed size without dict: {}", zstd::bulk::compress(&tokens_to_bytes(start.to_vec()), 3).unwrap().len());
    //    let decompressed = decompress_to_tokens(&compressed);
    //
    //    assert_eq!(decompressed, start);
    //}
}