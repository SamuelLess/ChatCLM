use crate::chat::ChatHistory;
use crate::component::prompt_input::PromptInput;
use crate::model::{cut_prompt, get_next_token};
use leptos::{
    component, spawn_local, view, Callback, IntoView, ReadSignal, SignalUpdate, WriteSignal,
};

#[component]
pub fn PromptSection(
    set_chat: WriteSignal<ChatHistory>,
    selected_model_index: ReadSignal<usize>,
) -> impl IntoView {
    view! {
        <section class="prompt">
            <div class="prompt__wrapper">
                <PromptInput on_submit=Callback::new(move |prompt: String| {
                    let model_idx = selected_model_index();
                    set_chat
                        .update(|chat| {
                            chat.new_user_message(prompt.clone());
                        });
                    spawn_local(async move {
                        set_chat
                            .update(|chat| {
                                chat.new_server_message("thinking...".to_string());
                            });
                        if let Ok(Some(response)) = get_next_token(model_idx, prompt.clone()).await
                        {
                            set_chat
                                .update(|chat| {
                                    chat.replace_last_server_message(
                                        cut_prompt(&prompt, &response),
                                    );
                                });
                            let mut response = response;
                            while let Ok(Some(stream)) = get_next_token(model_idx, response.clone())
                                .await
                            {
                                set_chat
                                    .update(|chat| {
                                        chat.replace_last_server_message(
                                            cut_prompt(&prompt, &stream),
                                        );
                                    });
                                response = stream;
                            }
                        }
                    });
                })/>

                <p class="prompt__disclaimer">ChatCLM can make mistakes. Check important info.</p>
            </div>
        </section>
    }
}
