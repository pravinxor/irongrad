mod layer;
mod neuron;
mod value;
use layer::Layer;
use neuron::Neuron;
use value::Value;

fn main() {
    let x = vec![2.0, 2.0];
    let n = Layer::new(2, 3);
    dbg!(n.call(&x));
}
