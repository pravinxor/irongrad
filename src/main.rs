mod layer;
mod mlp;
mod neuron;
mod value;
use layer::Layer;
use mlp::MLP;
use neuron::Neuron;
use value::Value;

fn main() {
    let xs: Vec<Vec<Value>> = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    .iter()
    .map(|x| x.iter().copied().map(Value::new).collect())
    .collect(); // 4 possible inputs

    let ys: Vec<Value> = [1.0, -1.0, -1.0, 1.0]
        .iter()
        .copied()
        .map(Value::new)
        .collect(); // 4 desired targets

    let n = MLP::new(3, &[4, 4, 1]);

    for i in 0..10 {
        let ypred: Vec<Value> = xs.iter().flat_map(|x| n.call(x)).collect();
        let mut loss = Value::new(0.0);
        for l in ypred
            .iter()
            .cloned()
            .zip(ys.iter().cloned())
            .map(|(yout, ygt)| (yout - ygt).powf(2.0))
        {
            loss = loss + l;
        }
        println!("Loss {}: {}", i, loss.inner.borrow().data);

        loss.backwards();
        n.parameters().iter().for_each(|p| {
            let mut v = p.inner.borrow_mut();
            v.data -= 0.0001 * v.grad;
        });
    }
}
