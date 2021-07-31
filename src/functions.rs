use std::cmp::Ordering;
use ndarray::prelude::*;
use ndarray::Array;

fn softmax<D: ndarray::Dimension>(x: &mut Array<f64, D>) -> Array<f64, D> {
    let comparer = |x, y| {
        if x < y { 
            Ordering::Less
        } else if x > y {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    };
    let max = x.iter().max_by(comparer);
    if let Some(max) = max.cloned() {
        let c = x.iter_mut().map(|e| *e - max);
    }
    for i in x.iter() {
    }
    //let sub = x - max;
    x.clone()
}
// def softmax(x):
//     e_x = np.exp(x - np.max(x))
//     return e_x / e_x.sum()

// fn cross_entropy_error(y: ndarray<f32>, t: ndarray<f32>) -> f32 {
//     if y.ndim == 1 {
//         t = t.reshape(1, t.size);
//         y = y.reshape(1, y.size);
//     }
//     if t.size == y.size {
//         t = t.argmax(axis=1);
//     }
//     let batch_size = y.shape[0];
//     -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
// }