use leptos::{component, create_signal, IntoView, view};
use crate::component::dropdown::Dropdown;

#[derive(Copy, Clone)]
enum Model {
    ChatCLM1_0,
    ChatGPT3_5,
    ChatGPT4o,
}

impl Model {
    pub fn name(&self) -> &'static str {
        match self {
            Model::ChatCLM1_0 => "ChatCLM 1.0",
            Model::ChatGPT3_5 => "ChatGPT 3.5",
            Model::ChatGPT4o => "ChatGPT 4o",
        }
    }
}

#[component]
pub fn NavBar() -> impl IntoView {
    let (selected_model_index, set_selected_model_index) = create_signal(0usize);

    view! {
        <nav class="navbar">
            <Dropdown
                options=[
                    Model::ChatCLM1_0.name(),
                    Model::ChatGPT3_5.name(),
                    Model::ChatGPT4o.name(),
                ]
                selected_option_index=selected_model_index
                set_selected_option_index=set_selected_model_index
            />
        </nav>
    }
}
