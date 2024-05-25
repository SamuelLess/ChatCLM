use crate::chat::Message;
use leptos::{component, view, IntoView, Show};

#[component]
pub fn ChatMessage(msg: Message) -> impl IntoView {
    let is_user_msg = msg.is_user_msg();

    view! {
        <div class="chat_message" class=("chat_message--machine", move || !is_user_msg)>
            <Show when=move || !is_user_msg>
                <div class="chat_message__icon">
                    <div>*</div>
                </div>
            </Show>

            <p class=("chat_message__bubble", move || is_user_msg)>{msg.message}</p>
        </div>
    }
}
