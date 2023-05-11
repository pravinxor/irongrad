mod neuron;
mod value;
use neuron::Neuron;
use value::Value;

fn main() {
    let x = vec![2.0, 2.0];
    let n = Neuron::new(2);
    dbg!(n.call(&x));
}
