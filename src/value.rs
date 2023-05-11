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
}

impl Operation {
    fn backward(&self, grad: f64) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Operation::Add(lhs, rhs) => {
                lhs.try_borrow_mut()?.grad += 1.0 * grad;
                rhs.try_borrow_mut()?.grad += 1.0 * grad;
                if let Some(op) = &lhs.borrow().op {
                    op.backward(lhs.borrow().grad)?;
                }
                if let Some(op) = &lhs.borrow().op {
                    op.backward(lhs.borrow().grad)?;
                }
            }
            Operation::Mul(lhs, rhs) => {
                lhs.try_borrow_mut()?.grad = rhs.borrow().data * grad;
                rhs.try_borrow_mut()?.grad += lhs.borrow().data * grad;
                if let Some(op) = &lhs.borrow().op {
                    op.backward(lhs.borrow().grad)?;
                }
                if let Some(op) = &rhs.borrow().op {
                    op.backward(rhs.borrow().grad)?;
                }
            }
            Operation::Tanh(v) => {
                let data = v.borrow().data;
                v.try_borrow_mut()?.grad += data * data;
                if let Some(op) = &v.borrow().op {
                    op.backward(v.borrow().grad)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct InnerValue {
    /// The value stored
    pub data: f64,

    /// The derivative of the current Value with respect to its @prev values
    grad: f64,

    /// The operation that created the value (None if the value was initialized with Self::new())
    op: Option<Operation>,
}

impl InnerValue {
    pub fn backwards(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.grad = 1.0;
        if let Some(op) = &self.op {
            op.backward(self.grad)?;
        }
        Ok(())
    }
}

impl PartialEq for InnerValue {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for InnerValue {}

impl std::hash::Hash for InnerValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(self, state);
    }
}

#[derive(Debug)]
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
}

impl std::ops::Add<Value> for Value {
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

impl std::ops::Mul<Value> for Value {
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
