pub struct MLP {
    layers: Vec<crate::Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let nin = vec![nin];
        let sz = nin
            .iter()
            .chain(nouts.iter())
            .zip(nin.iter().chain(nouts.iter()).skip(1));

        Self {
            layers: sz.map(|(i, j)| crate::Layer::new(*i, *j)).collect(),
        }
    }

    pub fn call(&self, x: &[crate::Value]) -> Vec<crate::Value> {
        let mut x = x.to_owned();
        for layer in &self.layers {
            x = layer.call(&x);
        }
        x
    }
}
