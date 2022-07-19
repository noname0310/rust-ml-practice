mod functions;

use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::view::ContinuousView;
use plotlib::style::{LineStyle};

fn make_plot<T: Into<String>>(vec: &Vec<f64>, f: fn(v: &Vec<f64>) -> Vec<f64>, colour: T) -> Plot {
    Plot::new(
        vec.iter()
        .zip(f(&vec))
        .map(|(x, y)| (x.clone(), y.clone()))
        .collect()
    ).line_style(
    LineStyle::new()
        .colour(colour),
    )
}

fn main() {
    let range_start = -80;
    let range_end = 80;
    let vec: Vec<f64> = (range_start..range_end).into_iter().map(|x| x as f64 / 10.).collect();

    let v = ContinuousView::new()
        .add(make_plot(&vec, functions::sigmoid_vec, "#DD3355"))
        .add(make_plot(&vec, functions::relu_vec, "#55DD33"))
        .add(make_plot(&vec, functions::step_vec, "#0000AA"))
        //.add(make_plot(&vec, functions::softmax_vec, "#00AA00"))
        .x_range((range_start / 10) as f64, (range_end / 10) as f64)
        .y_range(0., 1.)
        .x_label("x")
        .y_label("y");

    Page::single(&v).save("scatter.svg").unwrap();
}
