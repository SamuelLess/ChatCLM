use leptos::{component, view, Callback, IntoView, Show, NodeRef, html, create_node_ref, create_signal};

#[component]
pub fn PromptInput(on_submit: Callback<String>) -> impl IntoView {
    let (show_placeholder, set_show_placeholder) = create_signal(true);
    let textarea_element: NodeRef<html::Div> = create_node_ref();

    let submit = move || {
        let input = textarea_element().unwrap().inner_text().trim().to_string();
        if input.len() == 0 {
            return;
        }

        on_submit(input);
        textarea_element().unwrap().set_inner_html("");
    };

    view! {
        <div class="prompt_input">
            <div class="prompt_input__textarea_wrapper">
                <div
                    class="prompt_input__textarea"
                    contenteditable
                    node_ref=textarea_element
                    on:focus=move |_| {
                        set_show_placeholder(false);
                    }

                    on:blur=move |_| {
                        set_show_placeholder(textarea_element().unwrap().inner_text().len() == 0);
                    }

                    on:keydown=move |ev| {
                        if ev.key_code() == 13 && !ev.shift_key() {
                            ev.prevent_default();
                            submit();
                        }
                    }
                >
                </div>

                <Show when=move || show_placeholder()>
                    <div class="prompt_input__textarea_placeholder">Message ChatCLM</div>
                </Show>

                <button on:click=move |_| submit() class="prompt_input__send_button">
                    >
                </button>
            </div>
        </div>
    }
}
