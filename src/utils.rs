use std::sync::atomic::AtomicUsize;

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

#[inline]
pub fn float_eq(lhs: f64, rhs: f64) -> bool {
    f64::abs(lhs - rhs) < f64::EPSILON
}

#[inline]
pub fn float_approx(lhs: f64, rhs: f64, tol: f64) -> bool {
    f64::abs(lhs - rhs) < tol
}

static COUNTER: AtomicUsize = AtomicUsize::new(1);

pub struct IDManager {

}

impl IDManager {
    pub fn make_id() -> usize {
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

#[test]
fn float_eq_close_enough(){
    assert!(float_eq(1000., 1000.0));
    assert!(float_eq(1000., 1000.000000000000000000000000000000001));
}

#[test]
fn float_eq_not_equal() {
    assert!(!float_eq(1000., 1000.001));
}