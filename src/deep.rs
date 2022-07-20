use ndarray::{Array1, Array2, array, Array, Dimension};

pub fn max<D: ndarray::Dimension>(x: &Array<f64, D>) -> f64 {
    x.iter().fold(std::f64::MIN, |acc, &x| if x > acc { x } else { acc })
}

fn softmax<D: Dimension>(x: &Array<f64, D>) -> Array<f64, D> {
    if x.dim() == 2 {
        let mut x = x.t();
        x = x - max(&x);
        let y = x.mapv(|x| x.exp()) / x.mapv(|x| x.exp().sum());
        y.t()
    } else {
        let mut x = x - max(&x);
        let y = x.mapv(|x| x.exp()) / x.mapv(|x| x.exp().sum());
        y
    }
}
