use crate::chat::ChatHistory;
use crate::component::chat::Chat;
use crate::component::navbar::NavBar;
use crate::component::prompt_section::PromptSection;
use crate::error_template::{AppError, ErrorTemplate};
use leptos::*;
use leptos_meta::*;
use leptos_router::*;

#[component]
pub fn App() -> impl IntoView {
    // Provides context that manages stylesheets, titles, meta tags, etc.
    provide_meta_context();

    // create chat as reactive signal object
    let (chat, set_chat) = create_signal(ChatHistory::default());
    // fill with dummy data
    set_chat.update(|chat| {
        chat.new_server_message("Welcome to ChatCLM!".to_string());
        chat.new_user_message("Hello!".to_string());
        chat.new_server_message("Type a message and press Enter to chat.".to_string());
    });

    view! {
        <Stylesheet id="leptos" href="/pkg/chatclm.css"/>

        // sets the document title
        <Title text="ChatCLM"/>

        // content for this welcome page
        <Router fallback=|| {
            let mut outside_errors = Errors::default();
            outside_errors.insert_with_default_key(AppError::NotFound);
            view! { <ErrorTemplate outside_errors/> }.into_view()
        }>
            <main>
                <Routes>
                    <Route path="" view=move || view! { <HomePage chat=chat set_chat=set_chat/> }/>
                </Routes>
            </main>
        </Router>
    }
}

#[component]
fn HomePage(chat: ReadSignal<ChatHistory>, set_chat: WriteSignal<ChatHistory>) -> impl IntoView {
    view! {
        <NavBar/>

        <Chat chat=chat/>

        <PromptSection set_chat=set_chat/>
    }
}
