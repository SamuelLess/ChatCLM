use std::fmt::{Display, Formatter};
use std::str::FromStr;
use leptos::{Callback, component, create_signal, IntoView, Signal, view};
use crate::component::dropdown::Dropdown;

#[derive(Clone)]
enum Model {
    ChatCLM1_0,
    ChatGPT3_5,
    ChatGPT4o,
}

impl Display for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::ChatCLM1_0 => write!(f, "ChatCLM 1.0"),
            Model::ChatGPT3_5 => write!(f, "ChatGPT 3.5"),
            Model::ChatGPT4o => write!(f, "ChatGPT 4o"),
        }
    }
}

impl FromStr for Model {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ChatCLM 1.0" => Ok(Model::ChatCLM1_0),
            "ChatGPT 3.5" => Ok(Model::ChatGPT3_5),
            "ChatGPT 4o" => Ok(Model::ChatGPT4o),
            &_ => Err(()),
        }
    }
}

#[component]
pub fn NavBar() -> impl IntoView {
    let (selected_model, set_selected_model) = create_signal(Model::ChatCLM1_0);

    view! {
        <nav class="navbar">
            <Dropdown
                options=vec![
                    Model::ChatCLM1_0.to_string(),
                    Model::ChatGPT3_5.to_string(),
                    Model::ChatGPT4o.to_string(),
                ]
                selected_option=Signal::derive( move || selected_model().to_string())
                set_selected_option=Callback::from(move |option: String| set_selected_model(Model::from_str(&option).unwrap()))
            />
        </nav>
    }
}
