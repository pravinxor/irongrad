use std::rc::Rc;

#[derive(Debug)]
pub enum Operation {
    Add,
    Mul,
}

#[derive(Debug)]
/// Struct holding the lhs and rhs values that were combined with an operation
pub struct Previous<T> {
    lhs: Rc<InnerValue<T>>,
    rhs: Rc<InnerValue<T>>,
}

#[derive(Debug)]
pub struct InnerValue<T> {
    /// The value stored
    pub data: T,
    /// The operation that created the value (None if the value was initialized with Self::new())
    op: Option<Operation>,
    // The previous operations that created the value (Also None if the value was initialized with Self::new())
    prev: Option<Previous<T>>,
}

#[derive(Debug)]
pub struct Value<T> {
    pub inner: Rc<InnerValue<T>>,
}

impl<T> Value<T> {
    pub fn new(data: T) -> Self {
        Self {
            inner: Rc::new(InnerValue {
                data,
                op: None,
                prev: None,
            }),
        }
    }
}

impl<T: std::ops::Add<Output = T> + Copy> std::ops::Add<Value<T>> for Value<T> {
    type Output = Value<T>;
    fn add(self, rhs: Value<T>) -> Self::Output {
        Self::Output {
            inner: Rc::new(InnerValue {
                data: self.inner.data + rhs.inner.data,
                op: Some(Operation::Add),
                prev: Some(Previous {
                    lhs: self.inner,
                    rhs: rhs.inner,
                }),
            }),
        }
    }
}

impl<T: std::ops::Mul<Output = T> + Copy> std::ops::Mul<Value<T>> for Value<T> {
    type Output = Value<T>;
    fn mul(self, rhs: Value<T>) -> Self::Output {
        Self::Output {
            inner: Rc::new(InnerValue {
                data: self.inner.data * rhs.inner.data,
                op: Some(Operation::Mul),
                prev: Some(Previous {
                    lhs: self.inner,
                    rhs: rhs.inner,
                }),
            }),
        }
    }
}
