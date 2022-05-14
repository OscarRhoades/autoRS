pub mod network;
pub mod data;


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


    // let cost = network_main.cost(3);
    // println!("{}", cost);
    
}
