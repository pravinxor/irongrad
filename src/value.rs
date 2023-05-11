#[derive(Debug)]
/// An operation identifier, containing the enums that were used
pub enum Operation {
    Add(std::rc::Rc<InnerValue>, std::rc::Rc<InnerValue>),
    Mul(std::rc::Rc<InnerValue>, std::rc::Rc<InnerValue>),
    Tanh(std::rc::Rc<InnerValue>),
}

#[derive(Debug)]
/// Struct holding the lhs and rhs values that were combined with an operation
pub struct Previous {
    lhs: Option<std::rc::Rc<InnerValue>>,
    rhs: Option<std::rc::Rc<InnerValue>>,
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

impl InnerValue {
    /// Recursively builds the topological graph to be used for back-propogation
    pub fn build_topo<'a: 'b, 'b>(
        &'a self,
        topo: &mut Vec<&'a InnerValue>,
        visited: &mut std::collections::HashSet<&'b InnerValue>,
    ) {
        if !visited.contains(&self) {
            visited.insert(self);
            if let Some(op) = self.op.as_ref() {
                match op {
                    Operation::Add(rhs, lhs) | Operation::Mul(rhs, lhs) => {
                        rhs.build_topo(topo, visited);
                        lhs.build_topo(topo, visited);
                    }
                    Operation::Tanh(v) => v.build_topo(topo, visited),
                }
            }
            topo.push(self);
        }
    }
}

#[derive(Debug)]
pub struct Value {
    pub inner: std::rc::Rc<InnerValue>,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self {
            inner: std::rc::Rc::new(InnerValue {
                data,
                grad: 0.0, // Initialize to default/0 signifying no effect on gradient
                op: None,
            }),
        }
    }

    pub fn tanh(self) -> Self {
        let x = self.inner.data;
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0); // The actual tanh computation
        Self {
            inner: std::rc::Rc::new(InnerValue {
                data: t,
                grad: 0.0,
                op: Some(Operation::Tanh(self.inner)),
            }),
        }
    }

    pub fn build_topo(&self) -> Vec<&InnerValue> {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.inner.build_topo(&mut topo, &mut visited);
        topo
    }
}

impl std::ops::Add<Value> for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Self::Output {
        Self::Output {
            inner: std::rc::Rc::new(InnerValue {
                data: self.inner.data + rhs.inner.data,
                grad: 0.0,
                op: Some(Operation::Add(self.inner, rhs.inner)),
            }),
        }
    }
}

impl std::ops::Mul<Value> for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Self::Output {
        Self::Output {
            inner: std::rc::Rc::new(InnerValue {
                data: self.inner.data * rhs.inner.data,
                grad: 0.0,
                op: Some(Operation::Mul(self.inner, rhs.inner)),
            }),
        }
    }
}
