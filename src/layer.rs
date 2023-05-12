pub struct Layer {
    neurons: Vec<crate::Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        Layer {
            neurons: (0..nout).map(|_| crate::Neuron::new(nin)).collect(),
        }
    }

    pub fn call(&self, x: &[crate::Value]) -> Vec<crate::Value> {
        self.neurons.iter().cloned().map(|n| n.call(x)).collect()
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ ")?;
        for neuron in &self.neurons {
            write!(f, "{} ", neuron)?;
        }
        write!(f, "]")
    }
}
