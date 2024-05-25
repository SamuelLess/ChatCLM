use crate::chat::ChatHistory;
use crate::component::prompt_input::PromptInput;
use leptos::{component, view, Callback, IntoView, SignalUpdate, WriteSignal};

#[component]
pub fn PromptSection(set_chat: WriteSignal<ChatHistory>) -> impl IntoView {
    view! {
        <section class="prompt">
            <div class="prompt__wrapper">
                <PromptInput on_submit=Callback::new(move |prompt: String| {
                    set_chat
                        .update(|chat| {
                            chat.new_user_message(prompt);
                        });
                })/>

                <p class="prompt__disclaimer">ChatCLM can make mistakes. Check important info.</p>
            </div>
        </section>
    }
}
