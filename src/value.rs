#[derive(Debug)]
pub enum Operation {
    Add,
    Mul,
    Tanh,
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

    /// The previous value(s) that created the current value
    previous: Vec<std::rc::Rc<InnerValue>>,
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
    pub fn build_topo<'a: 'b, 'b>(
        &'a self,
        topo: &mut Vec<&'a InnerValue>,
        visited: &mut std::collections::HashSet<&'b InnerValue>,
    ) {
        if !visited.contains(&self) {
            visited.insert(self);
            self.previous
                .iter()
                .for_each(|child| child.build_topo(topo, visited));
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
                previous: vec![],
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
                op: Some(Operation::Tanh),
                previous: vec![self.inner],
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
                op: Some(Operation::Add),
                previous: vec![self.inner, rhs.inner],
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
                op: Some(Operation::Mul),
                previous: vec![self.inner, rhs.inner],
            }),
        }
    }
}
