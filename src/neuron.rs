use rand::Rng;

use crate::value::Value;

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(n_inputs: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        Neuron {
            weights: (0..n_inputs)
                .map(|_| rng.gen_range(-1.0..1.0))
                .map(Value::new)
                .collect(),
            bias: Value::new(rng.gen_range(-1.0..1.0)),
        }
    }
    pub fn call(self, x: &[f64]) -> Value {
        let mut act = self.bias;
        for z in self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(wi, xi)| (wi.clone(), Value::new(*xi)))
            .map(|(wi, xi)| wi * xi)
        {
            act = act + z;
        }
        act.tanh()
    }
}
