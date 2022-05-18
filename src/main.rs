pub mod network;
pub mod data;


use ndarray::prelude::*;
use ndarray::{concatenate, Axis, stack};

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Ix1;

use ndarray::{arr1, arr2, array};
use std::error::Error;

use ndarray::parallel::prelude::*;

use rand::{thread_rng, Rng};







fn main() {
    
    const DATA_ITEM: usize = 950;
    const ROW_SIZE: usize = 28;
    let data = data::generate(DATA_ITEM as i32);
    data::show(DATA_ITEM as i32);

    let mut network_main = network::Network::new();


    network_main.randomize_layers();
    network_main.mount(data.0, ROW_SIZE);
    

    network_main.forward_calculate();

    network_main.initial_backpropagate(data.1 as usize);


}
