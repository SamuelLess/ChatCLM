use leptos::{Callback, component, create_signal, For, IntoView, Signal, SignalSet, SignalUpdate, view};

#[component]
pub fn Dropdown(
    options: Vec<String>,
    selected_option: Signal<String>,
    set_selected_option: Callback<String>,
) -> impl IntoView {
    let (is_open, set_open) = create_signal(false);

    view! {
        <div class="dropdown">
            <div class="dropdown__selected" on:click=move |_| set_open.update(|is_currently_open| *is_currently_open = !*is_currently_open)>
                <div>{selected_option}</div>
                <div class="dropdown__icon">></div>
            </div>
            <div class="dropdown__options" class=("hidden", move || !is_open())>
                <For
                    each=move || options.clone()
                    key=|dropdown_option| dropdown_option.clone()
                    children=move |dropdown_option| {
                        let dropdown_option_clone= dropdown_option.clone();

                        view! {
                            <div class="dropdown__option" on:click=move |_| {
                                set_open.set(false);
                                set_selected_option(dropdown_option.clone());
                            }>
                                {dropdown_option_clone}
                            </div>
                        }
                    }
                />
            </div>
        </div>
    }
}
