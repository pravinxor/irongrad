mod value;
use value::Value;

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(-3.0);
    let c = Value::new(10.0);
    let e = a * b;
    let d = e + c;
    let f = Value::new(-2.0);
    let k = f.exp();
    let z = k.tanh();
    let l: Value = z * d;
    l.backwards();
    dbg!(&l);
}
