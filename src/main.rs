mod layer;
mod mlp;
mod neuron;
mod value;
use layer::Layer;
use mlp::MLP;
use neuron::Neuron;
use value::Value;

fn main() {
    let x: Vec<Value> = [2.0, 3.0, -1.0].iter().copied().map(Value::new).collect();
    let y = MLP::new(3, &vec![4, 4, 1]);
    for v in &y.call(&x) {
        println!("{}", v);
    }
}
