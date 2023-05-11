#[derive(Debug)]
pub struct Value<T> {
    pub data: T,
}

impl<T: std::ops::Add<Output = T>> std::ops::Add<Value<T>> for Value<T> {
    type Output = Value<T>;
    fn add(self, rhs: Value<T>) -> Self::Output {
        Self::Output {
            data: self.data + rhs.data,
        }
    }
}

impl<T: std::ops::Mul<Output = T>> std::ops::Mul<Value<T>> for Value<T> {
    type Output = Value<T>;
    fn mul(self, rhs: Value<T>) -> Self::Output {
        Self::Output {
            data: self.data * rhs.data,
        }
    }
}
