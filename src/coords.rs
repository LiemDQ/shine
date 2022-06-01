use std::ops::{Add, Sub, Neg, Mul, Div};
use std::cmp::{PartialEq};

#[cfg(test)]
use crate::assert_delta;


pub trait Coord {
    fn new(x: f64, y: f64, z: f64) -> Self;

    fn to_vec(self) -> Vector;

    fn to_pt(self) -> Point;

    fn zero() -> Self;
}

#[derive(Debug, Clone, Copy)]
pub struct Vector {
    pub x: f64,
    pub y: f64, 
    pub z: f64,
}

impl Vector {
    pub fn magnitude(&self) -> f64 {
        f64::sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    }

    pub fn normalize(self) -> Vector {
        let mag = self.magnitude();
        Self { x: self.x/mag, y: self.y/mag, z: self.z/mag}
    }

    pub fn reflect(&self, normal: &Vector) -> Vector {
        self - &((2.0*(self*normal)) * normal)
    }
}

impl Coord for Vector {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x: x, y: y, z: z }
    }

    fn to_vec(self) -> Vector {
        self
    }

    fn to_pt(self) -> Point {
        Point { x: self.x, y: self.y, z: self.z }        
    }

    fn zero() -> Self {
        Self { x: 0., y: 0., z: 0. }        
    }
}


impl Add for Vector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Sub for &Vector {
    type Output = Vector;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Vector {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<f64> for Vector {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vector> for Vector {
    type Output = f64;
    fn mul(self, rhs: Vector) -> Self::Output {
        self.x * rhs.x +
        self.y * rhs.y +
        self.z * rhs.z
    }
}

impl Mul<&Vector> for &Vector {
    type Output = f64;
    fn mul(self, rhs: &Vector) -> Self::Output {
        self.x * rhs.x +
        self.y * rhs.y +
        self.z * rhs.z
    }
}


impl Mul<Vector> for f64 {
    type Output = Vector;
    fn mul(self, rhs: Vector) -> Self::Output {
        Self::Output {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl Mul<&Vector> for f64 {
    type Output = Vector;
    fn mul(self, rhs: &Vector) -> Self::Output {
        Self::Output {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl Div<f64> for Vector {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl PartialEq for Vector {
    fn eq (&self, other: &Self) -> bool {
        f64::abs(self.x - other.x) < 0.00001 &&
        f64::abs(self.y - other.y) < 0.00001 &&
        f64::abs(self.z - other.z) < 0.00001
    }
}

impl PartialEq<Point> for Vector {
    fn eq(&self, _: &Point) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64, 
    pub z: f64,
}

impl Coord for Point {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x: x, y: y, z: z }
    }

    fn to_vec(self) -> Vector {
        Vector { x: self.x, y: self.y, z: self.z }
    }

    fn to_pt(self) -> Self {
        self
    }

    fn zero() -> Self {
        Self { x: 0., y: 0., z: 0. }        
    }
}

impl PartialEq for Point {
    fn eq (&self, other: &Self) -> bool {
        f64::abs(self.x - other.x) < 0.00001 &&
        f64::abs(self.y - other.y) < 0.00001 &&
        f64::abs(self.z - other.z) < 0.00001
    }
}

impl Add<Vector> for Point {
    type Output = Point;
    fn add(self, rhs: Vector) -> Self::Output {
        Self::Output {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub<Vector> for Point {
    type Output = Self;
    fn sub(self, rhs: Vector) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Sub<Point> for Point {
    type Output = Vector;
    fn sub(self, rhs: Point) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}


impl Sub<&Point> for &Point {
    type Output = Vector;
    fn sub(self, rhs: &Point) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}


fn dot(lhs: Vector, rhs: Vector) -> f64 {
    lhs * rhs
}


fn cross(lhs: Vector, rhs: Vector) -> Vector {
    Vector::new(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x *rhs.y - lhs.y*rhs.x)
}

// Tests
#[test]
fn create_point() {
    let pt = Point::new(4.3, -4.2, 3.1);
    assert_delta!(pt.x, 4.3, f64::EPSILON);
    assert_delta!(pt.y, -4.2, f64::EPSILON);
    assert_delta!(pt.z, 3.1, f64::EPSILON);
}

#[test]
fn add_coord() {
    let p1 = Point::new(3., -2., 5.);
    let p2 = Vector::new(-2.,3.,1.);
    let p3 = p1 + p2;
    assert_eq!(p3, Point::new(1.0, 1.0, 6.0));    
}

#[test]
fn sub_points() {
    let p1 = Point::new(3., 2., 1.);
    let p2 = Point::new(5., 6., 7.);
    let p3 = p1- p2;
    assert_eq!(p3, Vector::new(-2., -4., -6.));
}

#[test]
fn sub_vec_from_point(){
    let p = Point::new(3., 2., 1.);
    let v = Vector::new(5., 6., 7.);
    let f = p - v;
    assert_eq!(f, Point::new(-2., -4., -6.)); //now a point, still
}

#[test]
fn sub_vec_from_vec() {
    let v1 = Vector::new(3., 2., 1.);
    let v2 = Vector::new(5., 6.0, 7.);
    let f = v1 - v2;
    assert_eq!(f, Vector::new(-2., -4., -6.));
}

#[test]
fn mul_coord_with_scalar() {
    let v1 = Vector::new(1., -2., 3.);
    let v2 = v1.clone()*3.5;
    assert_eq!(v2, Vector::new(3.5, -7., 10.5));
    let v3 = v1*0.5; 
    assert_eq!(v3, Vector::new(0.5, -1., 1.5));
}

#[test]
fn div_coord_with_scalar() {
    let v1 = Vector::new(1., -2., 3.);
    let v2 = v1/2.0;
    assert_eq!(v2, Vector::new(0.5, -1., 1.5));

}

#[test]
fn vector_magnitude() {
    let v1 = Vector::new(1., 0., 0.);
    assert_delta!(v1.magnitude(), 1.0, f64::EPSILON);

    let v2 = Vector::new(0., 0., 1.);
    assert_delta!(v2.magnitude(), 1.0, f64::EPSILON);

    let v3 = Vector::new(1., 2., 3.);
    assert_delta!(v3.magnitude(), f64::sqrt(14.0), f64::EPSILON);

    let v4 = Vector::new(-1., -2., -3.);
    assert_delta!(v4.magnitude(), f64::sqrt(14.0), f64::EPSILON);
}

#[test]
fn normalize_vec() {
    let v1 = Vector::new(1., 2., 3.);
    let norm = f64::sqrt(14.0);
    assert_eq!(v1.normalize(), Vector::new(1./norm, 2./norm, 3./norm));
}

#[test]
fn dot_vec() {
    let v1 = Vector::new(1., 2., 3.);
    let v2 = Vector::new(2., 3., 4.);
    assert_eq!(v1*v2, 20.0)
}

#[test]
fn cross_product_vec() {
    let v1 = Vector::new(1., 2., 3.);
    let v2 = Vector::new(2., 3., 4.);
    assert_eq!(cross(v1.clone(), v2.clone()), Vector::new(-1., 2., -1.));
    assert_eq!(cross(v2, v1), Vector::new(1., -2., 1.));
    
}

#[test]
fn reflect_vector_at_45_deg() {
    let v = Vector::new(1., -1., 0.);
    let n = Vector::new(0., 1. ,0.);
    let r = v.reflect(&n);
    assert_eq!(r, Vector::new(1., 1., 0.));
}

#[test]
fn reflect_vector_off_slanted_surface(){
    let v = Vector::new(0., -1., 0.);
    let n = Vector::new(f64::sqrt(2.0)/2.0, f64::sqrt(2.0)/2.0 ,0.);
    let r = v.reflect(&n);
    assert_eq!(r, Vector::new(1., 0., 0.));
}