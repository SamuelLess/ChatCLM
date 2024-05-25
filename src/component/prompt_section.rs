use crate::chat::{cut_prompt, get_next_token, ChatHistory};
use crate::component::prompt_input::PromptInput;
use leptos::{
    component, spawn_local, view, Callback, IntoView, ReadSignal, SignalUpdate, WriteSignal,
};

#[component]
pub fn PromptSection(set_chat: WriteSignal<ChatHistory>) -> impl IntoView {
    view! {
        <section class="prompt">
            <div class="prompt__wrapper">
                <PromptInput on_submit=Callback::new(move |prompt: String| {
                    set_chat
                        .update(|chat| {
                            chat.new_user_message(prompt.clone());
                        });
                    spawn_local(async move {
                        set_chat
                            .update(|chat| {
                                chat.new_server_message("thinking...".to_string());
                            });
                        if let Ok(Some(response)) = get_next_token(prompt.clone()).await {
                            set_chat
                                .update(|chat| {
                                    chat.replace_last_server_message(
                                        cut_prompt(&prompt, &response),
                                    );
                                });
                            let mut response = response;
                            while let Ok(Some(stream)) = get_next_token(response.clone()).await {
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
