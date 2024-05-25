use leptos::{component, create_signal, For, IntoView, ReadSignal, SignalSet, SignalUpdate, view, WriteSignal};

#[component]
pub fn Dropdown<const N: usize>(
    options: [&'static str; N],
    selected_option_index: ReadSignal<usize>,
    set_selected_option_index: WriteSignal<usize>,
) -> impl IntoView {
    let (is_open, set_open) = create_signal(false);

    view! {
        <div class="dropdown">
            <div
                class="dropdown__selected"
                on:click=move |_| {
                    set_open.update(|is_currently_open| *is_currently_open = !*is_currently_open)
                }
            >

                <div>{move || options[selected_option_index()]}</div>
                <div class="dropdown__icon">></div>
            </div>
            <div class="dropdown__options" class=("hidden", move || !is_open())>
                <For
                    each=move || options.into_iter().enumerate()
                    key=|(_, option)| *option
                    children=move |(index, option)| {
                        view! {
                            <div
                                class="dropdown__option"
                                on:click=move |_| {
                                    set_open.set(false);
                                    set_selected_option_index.set(index);
                                }
                            >

                                {option}
                            </div>
                        }
                    }
                />

            </div>
        </div>
    }
}
