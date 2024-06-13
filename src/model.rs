use std::sync::{Arc, LazyLock, Once};
use leptos::{server, ServerFnError};
#[cfg(feature = "ssr")]
use tokio::sync::OnceCell;
#[cfg(feature = "ssr")]
use crate::backend::clm_model::ClmModel;

#[derive(Copy, Clone)]
pub enum Model {
    ChatCLM1_0,
    ChatGPT3_5,
    ChatGPT4o,
    ChatRandom,
}

impl Model {
    pub fn name(&self) -> &'static str {
        match self {
            Model::ChatCLM1_0 => "ChatCLM 0.1-pre-alpha",
            Model::ChatGPT3_5 => "ChatGPT 3.5",
            Model::ChatGPT4o => "ChatGPT 4o",
            Model::ChatRandom => "ChatRandom",
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Model::ChatCLM1_0,
            1 => Model::ChatGPT3_5,
            2 => Model::ChatGPT4o,
            3 => Model::ChatRandom,
            _ => panic!("Invalid model index"),
        }
    }


    #[cfg(feature = "ssr")]
    pub async fn predict_next_token(model_idx: usize, prompt: String) -> Option<String> {
        match Self::from_index(model_idx) {
            Model::ChatCLM1_0 => chat_clm_next_token(prompt).await,
            Model::ChatGPT4o => gpt4o_next_token(prompt).await,
            Model::ChatRandom => random_next_token(prompt).await,
            _ => random_next_token(prompt).await,
        }
    }
}

pub async fn random_next_token(prompt: String) -> Option<String> {
    // sleep 200 ms
    let random_number = rand::random::<u8>() % 7 + 1;
    if random_number == 1 {
        None
    } else {
        Some(format!("{} next", prompt))
    }
}

#[cfg(feature = "ssr")]
static CLM: LazyLock<ClmModel> = LazyLock::new(|| ClmModel::from_checkpoint("clm_model.bin"));

#[cfg(feature = "ssr")]
pub async fn chat_clm_next_token(prompt: String ) -> Option<String> {

    if prompt.len() > 250 {
        return None;
    }

    Some(CLM.predict_next(prompt, 1, 3))

}

pub async fn gpt4o_next_token(prompt: String) -> Option<String> {
    /*let client = Client::new();

    let request = CreateCompletionRequestArgs::default()
        .model("gpt-3.5-turbo-instruct")
        .prompt(prompt)
        .max_tokens(40_u16)
        .build().unwrap();

    let result = client.completions().create(request).await;
    if let Ok(response) = result {
        return Some(response.choices.clone()[0].clone().text);
    }*/
    None
}

#[server(GetNextToken, "/api")]
pub async fn get_next_token(
    model_idx: usize,
    prompt: String,
) -> Result<Option<String>, ServerFnError> {
    Ok(Model::predict_next_token(model_idx, prompt).await)
}

pub fn cut_prompt(prompt: &String, response: &String) -> String {
    let prompt = prompt.trim();
    let response = response.trim();
    if response.starts_with(prompt) {
        response[prompt.len()..].trim().to_string()
    } else {
        response.to_string()
    }
}
