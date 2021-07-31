use ndarray::prelude::*;

fn softmax(x: ndarray<f32>) -> ndarray<f32> {
    if x.ndim == 2 {
        x = x.T;
        x = x - np.max(x, axis=0);
        y = np.exp(x) / np.sum(np.exp(x), axis=0);
        return y.T;
    }
    x = x - np.max(x);
    np.exp(x) / np.sum(np.exp(x))
}

fn cross_entropy_error(y: ndarray<f32>, t: ndarray<f32>) -> f32 {
    if y.ndim == 1 {
        t = t.reshape(1, t.size);
        y = y.reshape(1, y.size);
    }
    if t.size == y.size {
        t = t.argmax(axis=1);
    }
    let batch_size = y.shape[0];
    -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
}