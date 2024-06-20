use itertools::Itertools;
use tiktoken_rs::{CoreBPE, p50k_base};
use tokenizers::tokenizer::Tokenizer;
use crate::backend::Token;

static TOKENIZER_PATH: &str = "tokenizer.json";

#[derive(Clone)]
pub enum ClmTokenizer {
    GPT2 (CoreBPE),
    Custom (Tokenizer),
}

impl ClmTokenizer {

    #[allow(dead_code)]
    fn new_gpt2() -> Self {
        let tokenizer = p50k_base().unwrap();
        ClmTokenizer::GPT2(tokenizer)
    }

    pub(crate) fn new_custom() -> Self {
        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        ClmTokenizer::Custom(tokenizer)
    }

    pub(crate) fn encode(&self, text: &str) -> Vec<Token> {
        match self {
            ClmTokenizer::GPT2(tokenizer) => tokenizer.encode_ordinary(text),
            ClmTokenizer::Custom(tokenizer) => {
                let encoding = tokenizer.encode(text, false).unwrap();
                let ids = encoding.get_ids();
                ids.iter().map(|&x| x as Token).collect()
            }
        }
    }

    pub(crate) fn decode(&self, tokens: Vec<Token>) -> String {
        match self {
            ClmTokenizer::GPT2(tokenizer) => tokenizer.decode(tokens).unwrap(),
            ClmTokenizer::Custom(tokenizer) => {
                tokenizer.decode(tokens.iter().map(|&x| x as u32).collect_vec().as_ref(), false).unwrap()
            }
        }
    }

}
#[cfg(test)]
mod tests {
    use crate::backend::tokenizer::{ClmTokenizer};

    #[test]
    fn custom_tokenizer_works() {
        let tokenizer = ClmTokenizer::new_custom();
        let encoding = tokenizer.encode("Hello, world!");
        let decoded = tokenizer.decode(encoding);
        assert_eq!(decoded, "hello, world!");
    }

    #[test]
    fn gpt2_tokenizer_works() {
        let tokenizer = ClmTokenizer::new_gpt2();
        let encoding = tokenizer.encode("Hello, world!");
        let decoded = tokenizer.decode(encoding);
        assert_eq!(decoded, "Hello, world!");
    }

    #[test]
    fn custom_tokenizer_special_characters() {
        let tokenizer = ClmTokenizer::new_custom();
        let encoding = tokenizer.encode("Hello, world! 🌍");
        let decoded = tokenizer.decode(encoding);
        assert_eq!(decoded, "hello, world! [UNK]");
    }
}

