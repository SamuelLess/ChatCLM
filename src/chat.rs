use leptos::{server, ServerFnError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sender {
    User,
    ChatCLM,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub message: String,
    pub time_iso: String,
    pub sender: Sender,
}

impl Message {
    pub fn is_user_msg(&self) -> bool {
        self.sender == Sender::User
    }

    pub fn new(message: String, sender: Sender) -> Self {
        Self {
            message,
            time_iso: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            sender,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChatHistory {
    pub messages: Vec<Message>,
}

impl ChatHistory {
    fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn new_sever_message(&mut self, message: String) {
        self.add_message(Message::new(message, Sender::ChatCLM));
    }

    pub fn new_user_message(&mut self, message: String) {
        self.add_message(Message::new(message, Sender::User));
    }

    pub fn append_to_last_server_message(&mut self, message: String) {
        if let Some(last_message) = self.messages.last_mut() {
            if last_message.sender == Sender::ChatCLM {
                last_message.message.push_str(&message);
            }
        }
    }
}

#[server(GetNextToken, "/api")]
pub async fn get_next_token(prompt: String) -> Result<String, ServerFnError> {
    Ok("Hello! I'm a message from the ChatCLM Server :)".to_string())
}
