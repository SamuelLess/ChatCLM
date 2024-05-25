use crate::chat::Message;
use leptos::{component, view, IntoView, Show};

#[component]
pub fn ChatMessage(msg: Message) -> impl IntoView {
    let user_msg = msg.user_msg();
    view! {
        <div class="chat_message" class=("chat_message--machine", move || user_msg)>
            <Show when=move || user_msg>
                <div class="chat_message__icon">
                    <div></div>
                </div>
            </Show>

            <p class=("chat_message__bubble", move || user_msg)>
                {msg.message}
            </p>
        </div>
    }
}
