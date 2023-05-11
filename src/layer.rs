pub struct Layer {
    neurons: Vec<crate::Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        Layer {
            neurons: (0..nout).map(|_| crate::Neuron::new(nin)).collect(),
        }
    }

    pub fn call(&self, x: &[f64]) -> Vec<crate::Value> {
        self.neurons
            .iter()
            .map(|n| n.to_owned())
            .map(|n| n.call(x))
            .collect()
    }
}
