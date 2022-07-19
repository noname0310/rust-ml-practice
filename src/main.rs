mod functions;

use std::ops::Range;

use plotters::{
    style::colors,
    prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea, LineSeries}
};

use ndarray::prelude::*;

fn draw_plot<F: FnMut(f64) -> f64>(
    f: F,
    x_range: Range<f64>,
    y_range: Range<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plot.png", (640, 480)).into_drawing_area();
    
    root.fill(&colors::WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(
            LineSeries::new(
                
                &colors::RED,
            )
        )?;

    root.present()?;

    Ok(())
}

fn main() {
    draw_plot(&functions::relu(&Array::from_vec(vec![-1.0, 0.0, 1.0]))).unwrap();
}
