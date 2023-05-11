#[derive(Debug)]
/// An operation identifier, containing the enums that were used
pub enum Operation {
    Add(
        std::rc::Rc<std::cell::RefCell<InnerValue>>,
        std::rc::Rc<std::cell::RefCell<InnerValue>>,
    ),
    Mul(
        std::rc::Rc<std::cell::RefCell<InnerValue>>,
        std::rc::Rc<std::cell::RefCell<InnerValue>>,
    ),
    Tanh(std::rc::Rc<std::cell::RefCell<InnerValue>>),
    Exp(std::rc::Rc<std::cell::RefCell<InnerValue>>),
    Pow(std::rc::Rc<std::cell::RefCell<InnerValue>>, f64),
}

impl Operation {
    fn backward(&self, grad: f64) {
        match self {
            Operation::Add(lhs, rhs) => {
                lhs.borrow_mut().grad += 1.0 * grad;
                rhs.borrow_mut().grad += 1.0 * grad;
                if let Some(op) = &lhs.borrow().op {
                    op.backward(lhs.borrow().grad);
                }
                if let Some(op) = &lhs.borrow().op {
                    op.backward(lhs.borrow().grad);
                }
            }
            Operation::Mul(lhs, rhs) => {
                lhs.borrow_mut().grad = rhs.borrow().data * grad;
                rhs.borrow_mut().grad += lhs.borrow().data * grad;
                if let Some(op) = &lhs.borrow().op {
                    op.backward(lhs.borrow().grad);
                }
                if let Some(op) = &rhs.borrow().op {
                    op.backward(rhs.borrow().grad);
                }
            }
            Operation::Tanh(v) => {
                let data = v.borrow().data;
                v.borrow_mut().grad += (1.0 - data.powi(2)) * grad;
                if let Some(op) = &v.borrow().op {
                    op.backward(v.borrow().grad);
                }
            }
            Operation::Exp(v) => {
                let data = v.borrow().data;
                v.borrow_mut().grad += data * grad;
                if let Some(op) = &v.borrow().op {
                    op.backward(v.borrow().grad);
                }
            }
            Operation::Pow(base, exponent) => {
                let data = base.borrow().data;
                base.borrow_mut().data += exponent * (data.powf(exponent - 1.0)) * grad;
                if let Some(op) = &base.borrow().op {
                    op.backward(base.borrow().grad);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct InnerValue {
    /// The value stored
    data: f64,

    /// The derivative of the current Value with respect to its @prev values
    grad: f64,

    /// The operation that created the value (None if the value was initialized with Self::new())
    op: Option<Operation>,
}

impl InnerValue {
    pub fn backwards(&mut self) {
        self.grad = 1.0;
        if let Some(op) = &self.op {
            op.backward(self.grad);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Value {
    pub inner: std::rc::Rc<std::cell::RefCell<InnerValue>>,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(InnerValue {
                data,
                grad: 0.0, // Initialize to default/0 signifying no effect on gradient
                op: None,
            })),
        }
    }

    pub fn tanh(self) -> Self {
        let x = self.inner.borrow().data;
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0); // The actual tanh computation
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(InnerValue {
                data: t,
                grad: 0.0,
                op: Some(Operation::Tanh(self.inner)),
            })),
        }
    }

    pub fn exp(self) -> Self {
        let x = self.inner.borrow().data;
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(InnerValue {
                data: x.exp(),
                grad: 0.0,
                op: Some(Operation::Exp(self.inner)),
            })),
        }
    }

    pub fn powf(self, n: f64) -> Self {
        let x = self.inner.borrow().data;
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(InnerValue {
                data: x.powf(n),
                grad: 0.0,
                op: Some(Operation::Pow(self.inner, n)),
            })),
        }
    }

    pub fn backwards(&self) {
        self.inner.borrow_mut().backwards();
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::new(value)
    }
}

impl std::ops::Add for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Self::Output {
        let sum = self.inner.borrow().data + rhs.inner.borrow().data;
        Self::Output {
            inner: std::rc::Rc::new(std::cell::RefCell::new(InnerValue {
                data: sum,
                grad: 0.0,
                op: Some(Operation::Add(self.inner, rhs.inner)),
            })),
        }
    }
}

impl std::ops::Neg for Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        self * Value::from(-1.0)
    }
}

impl std::ops::Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl std::ops::Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Self::Output {
        let product = self.inner.borrow().data * rhs.inner.borrow().data;
        Self::Output {
            inner: std::rc::Rc::new(std::cell::RefCell::new(InnerValue {
                data: product,
                grad: 0.0,
                op: Some(Operation::Mul(self.inner, rhs.inner)),
            })),
        }
    }
}

impl std::ops::Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.powf(-1.0)
    }
}
