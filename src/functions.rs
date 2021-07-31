use std::cmp::Ordering;
use ndarray::Array;

pub fn sigmoid_vec(x: &Vec<f64>) -> Vec<f64> {
    let mut sigmoid = x.clone();
    for (s, e) in sigmoid.iter_mut().zip(x) {
        *s = 1.0 / (1.0 + ((-e).exp2()));
    }
    sigmoid
}

pub fn softmax_vec(x: &Vec<f64>) -> Vec<f64> {
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
        result.iter_mut().for_each(|e| *e = (e.clone() - max).exp());
        let sum:f64 = result.iter().sum();
        result.iter_mut().for_each(|e| *e = e.clone() / sum);
    } else {
        result.iter_mut().for_each(|e| *e = e.exp());
        let sum:f64 = result.iter().sum();
        result.iter_mut().for_each(|e| *e = e.clone() / sum);
    };
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

pub fn numerical_gradient<D: ndarray::Dimension>(f, x) -> f64 {
    let h = 1e-4;
    let grad = np.zeros_like(x);
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while !it.finished {
        idx = it.multi_index;
        tmp_val = x[idx];
        x[idx] = float(tmp_val) + h;
        fxh1 = f(x);
        
        x[idx] = tmp_val - h;
        fxh2 = f(x);
        grad[idx] = (fxh1 - fxh2) / (2*h);
        
        x[idx] = tmp_val;
        it.iternext();
    }
    grad
}
