use crate::chat::ChatHistory;
use crate::component::chat_message::ChatMessage;
use leptos::{component, view, For, IntoView, ReadSignal};

#[component]
pub fn Chat(chat: ReadSignal<ChatHistory>) -> impl IntoView {
    view! {
        <section class="chat">
            <For
                each=move || chat().messages
                key=|it| it.time_iso.clone()
                children=move |it| {
            view! {
                <ChatMessage msg=it/>
            }
        }/>
        </section>
    }
}
