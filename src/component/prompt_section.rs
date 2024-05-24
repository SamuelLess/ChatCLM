use leptos::{component, IntoView, view};
use crate::component::prompt_input::PromptInput;

#[component]
pub fn PromptSection() -> impl IntoView {
    view! {
        <section class="prompt">
            <div class="prompt__wrapper">
                <PromptInput/>
            </div>
        </section>
    }
}
