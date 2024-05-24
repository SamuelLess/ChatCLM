use leptos::{component, IntoView, view};

#[component]
pub fn PromptInput() -> impl IntoView {
    view! {
        <div class="prompt_input">
            <div class="prompt_input__textarea_wrapper">
                // height = padding + line height * lines count = 20px + 25px * line count
                <textarea placeholder="Message ChatCLM" style="height: 45px" />

                <div class="prompt_input__send_button">></div>
            </div>
        </div>
    }
}
