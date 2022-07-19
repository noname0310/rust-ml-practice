use std::cmp::Ordering;
use ndarray::prelude::*;
use ndarray::{Array, Ix1};

pub fn relu<D: ndarray::Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    let mut result = x.clone();
    for i in result.iter_mut() {
        *i = if *i < 0.0 { 0.0 } else { *i }
    }
    result
}

pub fn step<D: ndarray::Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    let mut result = x.clone();
    for i in result.iter_mut() {
        *i = if *i < 0.0 { 0.0 } else { 1.0 }
    }
    result
}

pub fn sigmoid<D: ndarray::Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    let mut sigmoid = x.clone();
    for (s, e) in sigmoid.iter_mut().zip(x) {
        *s = 1.0 / (1.0 + ((-e).exp2()));
    }
    sigmoid
}

pub fn softmax<D: ndarray::Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
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

pub fn cross_entropy_error<D: ndarray::Dimension>(y: &Array<f64, D>, t: &Array<f64, D>) -> f64 {
    let delta = 1e-7;
    -(t * y.mapv(|x| (x + delta).log10())).sum()
}

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
