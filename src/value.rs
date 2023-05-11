use std::rc::Rc;

#[derive(Debug)]
pub enum Operation {
    Add,
    Mul,
}

#[derive(Debug)]
/// Struct holding the lhs and rhs values that were combined with an operation
pub struct Previous {
    lhs: Rc<InnerValue>,
    rhs: Rc<InnerValue>,
}

#[derive(Debug)]
pub struct InnerValue {
    /// The value stored
    pub data: f64,

    /// The derivative of the current Value with respect to its @prev values
    grad: f64,

    /// The operation that created the value (None if the value was initialized with Self::new())
    op: Option<Operation>,

    /// The previous operations that created the value (Also None if the value was initialized with Self::new())
    prev: Option<Previous>,
}

#[derive(Debug)]
pub struct Value {
    pub inner: Rc<InnerValue>,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self {
            inner: Rc::new(InnerValue {
                data,
                grad: 0.0, // Initialize to default/0 signifying no effect on gradient
                op: None,
                prev: None,
            }),
        }
    }
}

impl std::ops::Add<Value> for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Self::Output {
        Self::Output {
            inner: Rc::new(InnerValue {
                data: self.inner.data + rhs.inner.data,
                grad: 0.0,
                op: Some(Operation::Add),
                prev: Some(Previous {
                    lhs: self.inner,
                    rhs: rhs.inner,
                }),
            }),
        }
    }
}

impl std::ops::Mul<Value> for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Self::Output {
        Self::Output {
            inner: Rc::new(InnerValue {
                data: self.inner.data * rhs.inner.data,
                grad: 0.0,
                op: Some(Operation::Mul),
                prev: Some(Previous {
                    lhs: self.inner,
                    rhs: rhs.inner,
                }),
            }),
        }
    }
}
