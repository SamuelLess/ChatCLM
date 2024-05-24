use leptos::{component, IntoView, Show, view};

#[component]
pub fn ChatMessage(
    message: String,
    is_user_message: bool,
) -> impl IntoView {
    view! {
        <div class="chat_message" class=("chat_message--machine", move || !is_user_message)>
            <Show when=move || !is_user_message>
                <div class="chat_message__icon">
                    <div></div>
                </div>
            </Show>

            <p class=("chat_message__bubble", move || is_user_message)>
                {message}
            </p>
        </div>
    }
}
