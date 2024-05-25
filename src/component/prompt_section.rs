use leptos::{component, IntoView, view};
use crate::component::prompt_input::PromptInput;

#[component]
pub fn PromptSection() -> impl IntoView {
    view! {
        <section class="prompt">
            <div class="prompt__wrapper">
                <PromptInput/>

                <p class="prompt__disclaimer">ChatCLM can make mistakes. Check important info.</p>
            </div>
        </section>
    }
}
