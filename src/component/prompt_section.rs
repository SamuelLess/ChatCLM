use crate::chat::ChatHistory;
use crate::component::prompt_input::PromptInput;
use leptos::{component, view, IntoView, WriteSignal};

#[component]
pub fn PromptSection(set_chat: WriteSignal<ChatHistory>) -> impl IntoView {
    view! {
        <section class="prompt">
            <div class="prompt__wrapper">
                <PromptInput/>

                <p class="prompt__disclaimer">ChatCLM can make mistakes. Check important info.</p>
            </div>
        </section>
    }
}
