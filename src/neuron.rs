use rand::Rng;

#[derive(Debug, Clone)]
pub struct Neuron {
    weights: Vec<crate::Value>,
    bias: crate::Value,
}

impl Neuron {
    pub fn new(n_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        Neuron {
            weights: (0..n_inputs)
                .map(|_| rng.gen_range(-1.0..1.0))
                .map(crate::Value::new)
                .collect(),
            bias: crate::Value::new(rng.gen_range(-1.0..1.0)),
        }
    }
    pub fn call(self, x: &[f64]) -> crate::Value {
        let mut act = self.bias;
        for z in self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(wi, xi)| (wi.to_owned(), crate::Value::new(*xi)))
            .map(|(wi, xi)| wi * xi)
        {
            act = act + z;
        }
        act.tanh()
    }
}
