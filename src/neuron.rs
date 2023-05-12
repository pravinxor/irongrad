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
    pub fn call(self, x: &[crate::Value]) -> crate::Value {
        let mut act = self.bias;
        for z in self
            .weights
            .iter()
            .cloned()
            .zip(x.iter().cloned())
            .map(|(wi, xi)| wi * xi)
        {
            act = act + z;
        }
        act.tanh()
    }

    pub fn parameters(&self) -> Vec<crate::Value> {
        self.weights
            .iter()
            .cloned()
            .chain(vec![self.bias.clone()].iter().cloned())
            .collect()
    }
}

impl std::fmt::Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "W:")?;
        for weight in &self.weights {
            write!(f, "{} ", weight)?;
        }
        write!(f, " + B: {}", self.bias)
    }
}
