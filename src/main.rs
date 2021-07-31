mod functions;

use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::view::ContinuousView;
use plotlib::style::{LineStyle};

fn main() {
    let range_start = -80;
    let range_end = 80;
    let vec: Vec<f64> = (range_start..range_end).into_iter().map(|x| x as f64 / 10.).collect();

    let sigmoid_dataset = vec.iter()
        .zip(functions::sigmoid_vec(&vec))
        .map(|(x, y)| (x.clone(), y.clone()))
        .collect();
    
    let softmax_dataset = vec.iter()
        .zip(functions::softmax_vec(&vec))
        .map(|(x, y)| (x.clone(), y.clone()))
        .collect();

    let sigmoid_plot: Plot = Plot::new(sigmoid_dataset).line_style(
        LineStyle::new()
            .colour("#DD3355"),
    );

    let softmax_plot: Plot = Plot::new(softmax_dataset).line_style(
        LineStyle::new()
            .colour("#DD0055"),
    );

    let v = ContinuousView::new()
        .add(sigmoid_plot)
        .add(softmax_plot)
        .x_range((range_start / 10) as f64, (range_end / 10) as f64)
        .y_range(0., 1.)
        .x_label("x")
        .y_label("y");

    Page::single(&v).save("scatter.svg").unwrap();
}
