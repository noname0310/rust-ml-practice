use ndarray::prelude::*;
use ndarray::{Array, Ix1};

pub fn relu<D: ndarray::Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    x.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

pub fn step<D: ndarray::Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    x.mapv(|x| if x < 0.0 { 0.0 } else { 1.0 })
}

pub fn sigmoid<D: ndarray::Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

pub fn max<D: ndarray::Dimension>(x: &Array<f64, D>) -> f64 {
    x.iter().fold(std::f64::MIN, |acc, &x| if x > acc { x } else { acc })
}

pub fn softmax<D: ndarray::Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    let c = max(x);
    let exp_x = x.mapv(|x| (x - c).exp());
    let sum_exp_x = exp_x.sum();
    exp_x / sum_exp_x
}

#[allow(dead_code)]
pub fn cross_entropy_error<D: ndarray::Dimension>(y: &Array<f64, D>, t: &Array<f64, D>) -> f64 {
    let delta = 1e-7;
    -(t * y.mapv(|x| (x + delta).log10())).sum()
}

#[allow(dead_code)]
pub fn numerical_gradient<D: ndarray::Dimension>(
    f: fn(f64) -> f64, 
    x: &Array<f64, D>) -> Array<f64, Ix1> {
    let delta = 1e-5;
    let mut grad = Array::<f64, Ix1>::zeros((3).f());
    for (i, g) in x.iter().zip(&mut grad) {
        let fxh1 = f(*i + delta);
        let fxh2 = f(*i - delta);
        *g = (fxh1 - fxh2) / (2. * delta);
    }
    grad
}
