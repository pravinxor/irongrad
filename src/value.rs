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
