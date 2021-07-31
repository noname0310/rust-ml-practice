use std::cmp::Ordering;
use ndarray::Array;

fn softmax<D: ndarray::Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    let mut result = x.clone();
    let max = result.iter().max_by(|x, y| {
        if x < y { 
            Ordering::Less
        } else if x > y {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    if let Some(max) = max.cloned() {
        result.map_mut(|e| *e = (e.clone() - max).exp());
        let sum = result.sum();
        result.map_mut(|e| *e = e.clone() / sum);
    } else {
        result.map_mut(|e| *e = e.exp());
        let sum = result.sum();
        result.map_mut(|e| *e = e.clone() / sum);
    };
    result
}

fn cross_entropy_error<D: ndarray::Dimension>(y: &Array<f64, D>, t: &Array<f64, D>) -> f64 {
    let delta = 1e-7;
    -(t * y.mapv(|x| (x + delta).log10())).sum()
}
