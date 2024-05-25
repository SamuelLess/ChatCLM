use crate::component::dropdown::Dropdown;
use crate::model::Model;
use leptos::{component, create_signal, view, IntoView, ReadSignal, WriteSignal};

#[component]
pub fn NavBar(
    selected_model_index: ReadSignal<usize>,
    set_selected_model_index: WriteSignal<usize>,
) -> impl IntoView {
    view! {
        <nav class="navbar">
            <Dropdown
                options=[
                    Model::ChatCLM1_0.name(),
                    Model::ChatGPT3_5.name(),
                    Model::ChatGPT4o.name(),
                    Model::ChatRandom.name(),
                ]

                selected_option_index=selected_model_index
                set_selected_option_index=set_selected_model_index
            />
        </nav>
    }
}
