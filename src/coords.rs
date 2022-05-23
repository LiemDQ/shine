use std::ops::{Add, Sub, Neg, Mul, Div};
use std::cmp::{Eq, PartialEq};

/// Assert that `x` and `y` are within `d` of each other. 
/// Useful for checking floating-point values for equality.
macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if $x - $y > 0.0 {
            assert!($x - $y < $d);
        } else {
            assert!($y - $x < $d);
        }
    };
}

#[derive(Debug, Clone)]
struct Coord {
    pub x: f64,
    pub y: f64,
    pub z: f64, 
    pub is_pt: i32,
}

impl Coord {
    pub fn new(x: f64, y: f64, z: f64, w: i32) -> Self {
        Self { x: x, y: y, z: z, is_pt: w }
    }

    pub fn vec(x: f64, y: f64, z: f64) -> Self {
        Self { x: x, y: y, z: z, is_pt: 0 }
    }

    pub fn pt(x: f64, y: f64, z: f64) -> Self {
        Self { x: x, y: y, z: z, is_pt: 1 }
    }

    pub fn zero() -> Self {
        Self {x: 0.0, y: 0.0, z: 0.0, is_pt: 0}
    }

    pub fn magnitude(&self) -> f64 {
        assert_eq!(self.is_pt, 0);
        f64::sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    }

    pub fn normalize(self) -> Coord {
        assert_eq!(self.is_pt, 0);
        let mag = self.magnitude();
        Self { x: self.x/mag, y: self.y/mag, z: self.z/mag, is_pt: 0 }
    }
}

impl Add for Coord {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            is_pt: self.is_pt + rhs.is_pt,
        }
    }
}

impl Sub for Coord {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            is_pt: self.is_pt - rhs.is_pt
        }
    }
}

impl Neg for Coord {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            is_pt: -self.is_pt, //it is debatable whether this is the right thing to do, but in general points should not be negated.
        }
    }
}

impl Mul<f64> for Coord {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            is_pt: (self.is_pt as f64 * rhs) as i32,
        }
    }
}

impl Mul<Coord> for Coord {
    type Output = f64;
    fn mul(self, rhs: Coord) -> Self::Output {
        assert!(self.is_pt == 0 && rhs.is_pt == 0);
        self.x * rhs.x +
        self.y * rhs.y +
        self.z * rhs.z
    }
}

impl Div<f64> for Coord {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            is_pt: (self.is_pt as f64 / rhs) as i32,
        }
    }
}

impl PartialEq for Coord {
    fn eq (&self, other: &Self) -> bool {
        self.is_pt == other.is_pt &&
        f64::abs(self.x - other.x) < f64::EPSILON &&
        f64::abs(self.y - other.y) < f64::EPSILON &&
        f64::abs(self.z - other.z) < f64::EPSILON
    }
}

fn cross(lhs: Coord, rhs: Coord) -> Coord {
    assert!(lhs.is_pt == 0 && rhs.is_pt == 0);
    Coord::vec(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x *rhs.y - lhs.y*rhs.x)
}


#[test]
fn create_point() {
    let pt = Coord::new(4.3, -4.2, 3.1, 1);
    assert_delta!(pt.x, 4.3, f64::EPSILON);
    assert_delta!(pt.y, -4.2, f64::EPSILON);
    assert_delta!(pt.z, 3.1, f64::EPSILON);
    assert_eq!(pt.is_pt, 1);
}

#[test]
fn add_coord() {
    let p1 = Coord::new(3., -2., 5., 1);
    let p2 = Coord::new(-2.,3.,1., 0);
    let p3 = p1 + p2;
    assert_eq!(p3, Coord::new(1.0, 1.0, 6.0, 1));    
}

#[test]
fn sub_points() {
    let p1 = Coord::new(3., 2., 1., 1);
    let p2 = Coord::new(5., 6., 7., 1);
    let p3 = p1- p2;
    assert_eq!(p3, Coord::new(-2., -4., -6., 0));
}

#[test]
fn sub_vec_from_point(){
    let p = Coord::new(3., 2., 1., 1);
    let v = Coord::new(5., 6., 7., 0);
    let f = p - v;
    assert_eq!(f, Coord::new(-2., -4., -6., 1)); //now a point, still
}

#[test]
fn sub_vec_from_vec() {
    let v1 = Coord::new(3., 2., 1., 0);
    let v2 = Coord::new(5., 6.0, 7., 0);
    let f = v1 - v2;
    assert_eq!(f, Coord::new(-2., -4., -6., 0));
}

#[test]
fn mul_coord_with_scalar() {
    let v1 = Coord::new(1., -2., 3., -4);
    let v2 = v1.clone()*3.5;
    assert_eq!(v2, Coord::new(3.5, -7., 10.5, -14));
    let v3 = v1*0.5; 
    assert_eq!(v3, Coord::new(0.5, -1., 1.5, -2));
}

#[test]
fn div_coord_with_scalar() {
    let v1 = Coord::new(1., -2., 3., -4);
    let v2 = v1/2.0;
    assert_eq!(v2, Coord::new(0.5, -1., 1.5, -2));

}

#[test]
fn vector_magnitude() {
    let v1 = Coord::vec(1., 0., 0.);
    assert_delta!(v1.magnitude(), 1.0, f64::EPSILON);

    let v2 = Coord::vec(0., 0., 1.);
    assert_delta!(v2.magnitude(), 1.0, f64::EPSILON);

    let v3 = Coord::vec(1., 2., 3.);
    assert_delta!(v3.magnitude(), f64::sqrt(14.0), f64::EPSILON);

    let v4 = Coord::vec(-1., -2., -3.);
    assert_delta!(v4.magnitude(), f64::sqrt(14.0), f64::EPSILON);
}

#[test]
fn normalize_vec() {
    let v1 = Coord::vec(1., 2., 3.);
    let norm = f64::sqrt(14.0);
    assert_eq!(v1.normalize(), Coord::vec(1./norm, 2./norm, 3./norm));
}

#[test]
fn dot_vec() {
    let v1 = Coord::vec(1., 2., 3.);
    let v2 = Coord::vec(2., 3., 4.);
    assert_eq!(v1*v2, 20.0)
}

#[test]
fn cross_product_vec() {
    let v1 = Coord::vec(1., 2., 3.);
    let v2 = Coord::vec(2., 3., 4.);
    assert_eq!(cross(v1.clone(), v2.clone()), Coord::vec(-1., 2., -1.));
    assert_eq!(cross(v2, v1), Coord::vec(1., -2., 1.));
    
}