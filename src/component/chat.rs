use leptos::{component, IntoView, view};
use crate::component::chat_message::ChatMessage;

#[component]
pub fn Chat() -> impl IntoView {
    view! {
        <section class="chat">
            <ChatMessage message=String::from("test") is_user_message=true />
            <ChatMessage message=String::from("abcdef sjdf sdf djks") is_user_message=false />
        </section>
    }
}
