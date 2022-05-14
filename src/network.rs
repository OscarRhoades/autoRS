use ndarray::prelude::*;
use ndarray::{concatenate, Axis};

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Ix1;

use ndarray::{arr1, arr2, array};
use std::error::Error;

use ndarray::parallel::prelude::*;

use rand::{thread_rng, Rng};

struct WeightMatrix {
    matrix: ndarray::Array2<f32>,
    gradient: ndarray::Array2<f32>,
}

struct Biases {
    biases: ndarray::Array1<f32>,
    gradient: ndarray::Array1<f32>,
}

struct Activations {
    activations: ndarray::Array1<f32>,
    arguments: ndarray::Array1<f32>,
}

struct NetworkLayer {
    activations: Activations,
    weights: WeightMatrix,
    biases: Biases,
}

fn randomize_value(a: f32, b: f32) -> f32 {
    let mut rng = thread_rng();
    rng.gen_range(a..b)
}

impl NetworkLayer {
    fn new(layer_size: usize, prior_act: usize) -> NetworkLayer {
        NetworkLayer {
            activations: Activations {
                activations: ndarray::Array::zeros(layer_size),
                arguments: ndarray::Array::zeros(layer_size),
            },

            weights: WeightMatrix {
                matrix: ndarray::Array::zeros((layer_size, prior_act)),
                gradient: ndarray::Array::zeros((layer_size, prior_act)),
            },

            biases: Biases {
                biases: ndarray::Array::zeros(layer_size),
                gradient: ndarray::Array::zeros(layer_size),
            },
        }
    }

    fn randomize_layer(&mut self, a: f32, b: f32) {
        self.weights
            .matrix
            .par_map_inplace(|x| *x = randomize_value(a, b));
        self.biases
            .biases
            .par_map_inplace(|x| *x = randomize_value(a, b));
    }

    fn randomize_activations(&mut self, a: f32, b: f32) {
        self.activations
            .activations
            .par_map_inplace(|x| *x = randomize_value(a, b));
    }

    fn reset_activations(&mut self) {
        self.activations.activations.par_map_inplace(|x| *x = 0.0);
    }

    fn relu(&mut self) {
        self.activations
            .activations
            .par_map_inplace(|x| *x = if x > &mut 0.0 { *x } else { *x * 0.2 });
    }

    fn tanh(&mut self) {
        self.activations
            .activations
            .par_map_inplace(|x| *x = x.tanh());
    }

    fn softmax(&mut self) {
        let mut exp_vector = self.activations.activations.map(|x| (*x as f32).exp());
        // exp_vector.map(|x| println!("{}", x));

        let sum = exp_vector.sum();
        // println!("{}", sum);

        exp_vector.par_map_inplace(|x| *x = *x / sum);
        // println!("--------------------");
        // exp_vector.map(|x| println!("{}", x));
        // println!("softmax sum:{}", exp_vector.sum());

        self.activations.activations = exp_vector;
    }

    fn calculate_Z(&mut self, X: &ndarray::Array1<f32>) {
        let z_sum = self.weights.matrix.dot(X) + &self.biases.biases;

        //barrow checker trick.
        let activations_copy = z_sum.clone();
        self.activations.arguments = z_sum;
        self.activations.activations = activations_copy;
    }
}

const NETWORK_SIZE: usize = 4;
const INPUT_SIZE: usize = 784;
const ROW_SIZE: usize = 28;
const OUTPUT_SIZE: usize = 10;
const NETWORK_CONFIG: [usize; NETWORK_SIZE] = [INPUT_SIZE, 40, 20, OUTPUT_SIZE];

pub struct Network {
    network: Vec<NetworkLayer>,
    train_sessions: usize,
}

impl Network {
    pub fn new() -> Network {
        let mut network_build = Network {
            network: Vec::with_capacity(NETWORK_SIZE),
            train_sessions: 0,
        };

        let mut previous_config = 0;

        for config in NETWORK_CONFIG {
            network_build
                .network
                .push(NetworkLayer::new(config, previous_config));
            previous_config = config;
        }


        //these are the empty weights and biases in the inital layer
        network_build.network[0].weights.matrix = arr2(&[[]]);
        network_build.network[0].weights.gradient = arr2(&[[]]);

        network_build.network[0].biases.biases = arr1(&[]);
        network_build.network[0].biases.gradient = arr1(&[]);

        network_build
    }

    pub fn forward_calculate(&mut self) {
        //this function is a causality in the borrow-checker wars
        let mut previous = self.network[0].activations.activations.clone();

        for (index, layer) in self.network.iter_mut().enumerate() {
            if index != 0 {
                layer.calculate_Z(&previous);
                if index != NETWORK_SIZE - 1 {
                    layer.tanh()
                } else {
                    layer.softmax()
                };
                //inefficient clone
                previous = layer.activations.activations.clone();
            }
        }
    }

    pub fn cost(&self, answer: i8) -> f32 {
        let mut correct_distrobution = ndarray::Array1::<f32>::zeros(OUTPUT_SIZE);
        correct_distrobution[answer as usize] = 1.0;

        let mut cost_distrobution =
            correct_distrobution - &self.network[NETWORK_SIZE - 1].activations.activations;
        cost_distrobution.par_map_inplace(|x| *x = x.powf(2.0));
        cost_distrobution.sum() / (2.0 * OUTPUT_SIZE as f32)
    }

    pub fn initial_backpropagate(&mut self, answer: usize) {
        let mut correct_distrobution = ndarray::Array1::<f32>::zeros(OUTPUT_SIZE);
        correct_distrobution[answer] = 1.0;

        //cost with respect to first activation
        let initial_cost_derivative =
            (correct_distrobution - &self.network[NETWORK_SIZE - 1].activations.activations) * -1.0;

        //first activation with respect to z which is softmax
        let mut softmax_derivative = self.network[NETWORK_SIZE - 1]
            .activations
            .arguments
            .map(|x| (*x as f32).exp());
        let softmax_sum = softmax_derivative.sum();
        softmax_derivative.par_map_inplace(|x| *x = *x / softmax_sum);
        softmax_derivative.par_map_inplace(|x| *x = *x * (1.0 - *x));

        //combined with respect to cost
        let bias_product = softmax_derivative * initial_cost_derivative;
        //cost bias
        self.network[NETWORK_SIZE - 1].biases.gradient = bias_product; //assign to bias gradient

        //cost respect to weights
        let mut backward_activations = self.network[NETWORK_SIZE - 2]
            .activations
            .activations
            .clone();
        let mut weight_addition = self.network[NETWORK_SIZE - 1].weights.gradient.clone();
    }

    pub fn mount(&mut self, image: Array2<f32>, row_size: usize) {
        let mut linear_data = arr1(&[]);
        //takes the training 2d array and turns it into a linear array with all the elements
        for index in 0..row_size {
            linear_data = concatenate!(Axis(0), linear_data, image.slice(s![index, ..]));
        }

        // println!("{}", linear_data.len());
        // assert!(linear_data.len() == INPUT_SIZE);
        self.network[0].activations.activations = linear_data;
    }

    pub fn print_activations(&self) {
        for (i, layer) in self.network.iter().enumerate() {
            println!("{}: {}", i, layer.activations.activations)
        }
    }

    pub fn print_inputs(&self) {
        self.network[0]
            .activations
            .activations
            .map(|x| println!("{}", x));
    }

    pub fn randomize_layers(&mut self) {
        for layer in self.network.iter_mut() {
            layer.randomize_layer(-1.0, 1.0);
        }
    }
}
