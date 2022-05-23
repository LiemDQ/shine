/// Assert that `x` and `y` are within `d` of each other. 
/// Useful for checking floating-point values for equality.
#[macro_export]
macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if $x - $y > 0.0 {
            assert!($x - $y < $d);
        } else {
            assert!($y - $x < $d);
        }
    };
}

pub fn float_eq(lhs: f64, rhs: f64) -> bool {
    f64::abs(lhs - rhs) < f64::EPSILON
}