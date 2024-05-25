use leptos::{component, create_signal, event_target_value, view, IntoView};

#[component]
pub fn PromptInput() -> impl IntoView {
    let (input, set_name) = create_signal("".to_string());

    view! {
        <div class="prompt_input">
            <div class="prompt_input__textarea_wrapper">
                // height = padding + line height * lines count = 20px + 25px * line count
                <textarea type="text" style="height: 45px" placeholder="Message ChatCLM"
                    on:input=move |ev| {
                        set_name(event_target_value(&ev));
                    }
                    prop:value=input
                />

                <button class="prompt_input__send_button">></button>
            </div>
        </div>
    }
}
