use ndarray::{Array1, Array2, array, Array};

#[derive(Debug)]
struct Layer {
    perceptrons: Array2<f64>,
    biases: Array1<f64>,
    activation_function: fn(f64) -> f64,
}

impl Layer {
    pub fn new(
        perceptrons: Array2<f64>,
        biases: Array1<f64>,
        activation_function: fn(f64) -> f64
    ) -> Layer {
        Layer {
            perceptrons,
            biases,
            activation_function,
        }
    }

    pub fn forward(&self, x: &Array2<f64>, apply_activation: bool) -> Array2<f64> {
        let mut result = Array::from_iter( 
            x.dot(&self.perceptrons)
                .into_iter().zip(self.biases.iter())
                .map(|(x, b)| x + b)
        );

        if apply_activation {
            result.mapv_inplace(|x| (self.activation_function)(x));
        }

        result.into_shape((x.shape()[0], self.perceptrons.shape()[1])).unwrap()
    }
}

#[derive(Debug)]
struct Network {
    layers: Vec<Layer>
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Network {
        Network { layers }
    }

    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let x_len = x.len();
        let mut y = x.clone().into_shape((1, x_len)).unwrap();
        
        for i in 0..self.layers.len() - 1 {
            y = self.layers[i].forward(&y, true);
        }

        y = self.layers[self.layers.len() - 1].forward(&y, false);

        y.into_shape((x_len,)).unwrap()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn network_test() {
    let layer1 = Layer::new(
        array![ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6] ],
        array![0.1, 0.2, 0.3],
        sigmoid
    );

    let layer2 = Layer::new(
        array![ [0.1, 0.4], [0.2, 0.5], [0.3, 0.6] ],
        array![0.1, 0.2],
        sigmoid
    );
    
    let layer3 = Layer::new(
        array![ [0.1, 0.3], [0.2, 0.4] ],
        array![0.1, 0.2],
        sigmoid
    );

    let network = Network::new(vec![layer1, layer2, layer3]);

    let x = array![1.0, 0.5];
    let y = network.forward(&x);
    println!("{y}");
}
