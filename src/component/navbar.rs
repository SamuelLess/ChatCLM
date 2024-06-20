use crate::component::dropdown::Dropdown;
use crate::model::FrontendModel;
use leptos::{component, view, IntoView, ReadSignal, WriteSignal};

#[component]
pub fn NavBar(
    selected_model_index: ReadSignal<usize>,
    set_selected_model_index: WriteSignal<usize>,
) -> impl IntoView {
    view! {
        <nav class="navbar">
            <Dropdown
                options=[
                    FrontendModel::ChatCLM1_0.name(),
                    FrontendModel::ChatGPT3_5.name(),
                    FrontendModel::ChatGPT4o.name(),
                    FrontendModel::ChatRandom.name(),
                ]

                selected_option_index=selected_model_index
                set_selected_option_index=set_selected_model_index
            />
        </nav>
    }
}
