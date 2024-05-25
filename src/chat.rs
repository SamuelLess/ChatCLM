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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatHistory {
    pub messages: Vec<Message>,
}

impl Default for ChatHistory {
    fn default() -> Self {
        Self { messages: vec![] }
    }
}

impl ChatHistory {
    fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn new_sever_message(&mut self, message: String) {
        self.add_message(Message {
            message,
            time_iso: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            sender: Sender::ChatCLM,
        });
    }

    pub fn new_user_message(&mut self, message: String) {
        self.add_message(Message {
            message,
            time_iso: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            sender: Sender::User,
        });
    }
}
