use leptos::{component, create_signal, event_target_value, view, Callback, IntoView};

#[component]
pub fn PromptInput(on_submit: Callback<String>) -> impl IntoView {
    let (input, set_input) = create_signal("".to_string());

    view! {
        <div class="prompt_input">
            <div class="prompt_input__textarea_wrapper">
                // height = padding + line height * lines count = 20px + 25px * line count
                <textarea
                    type="text"
                    style="height: 45px"
                    placeholder="Message ChatCLM"
                    on:input=move |ev| {
                        set_input(event_target_value(&ev));
                    }

                    prop:value=input
                ></textarea>

                <button
                    on:click=move |_| {
                        on_submit(input().clone());
                        set_input("".to_string());
                    }

                    class="prompt_input__send_button"
                >
                    >
                </button>
            </div>
        </div>
    }
}
