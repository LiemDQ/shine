use crate::matrix::{LinAlg, Matrix4};


pub fn translation(x: f64, y: f64, z: f64) -> Matrix4 {
    let mut mat = Matrix4::ident();
    mat.set(x, 0, 3);
    mat.set(y, 1, 3);
    mat.set(z, 2, 3);
    mat
}

pub fn scaling(x: f64, y: f64, z: f64) -> Matrix4 {
    let mut mat = Matrix4::ident();
    mat.set(x, 0, 0);
    mat.set(y, 1, 1);
    mat.set(z, 2, 2);
    mat
}

/// create a rotation matrix4 around the x axis. The angle of rotation is specified in radians.
pub fn rotation_x(angle: f64) -> Matrix4 {
    let mut mat = Matrix4::ident();
    mat.set(angle.cos(), 1, 1);
    mat.set(angle.cos(), 2, 2);
    mat.set(-angle.sin(), 1, 2);
    mat.set(angle.sin(), 2, 1);
    mat
}

pub fn rotation_y(angle: f64) -> Matrix4 {
    let mut mat = Matrix4::ident();
    mat.set(angle.cos(), 0, 0);
    mat.set(angle.cos(), 2, 2);
    mat.set(-angle.sin(), 2, 0);
    mat.set(angle.sin(), 0, 2);
    mat
}

pub fn rotation_z(angle: f64) -> Matrix4 {
    let mut mat = Matrix4::ident();
    mat.set(angle.cos(), 0, 0);
    mat.set(angle.cos(), 1, 1);
    mat.set(-angle.sin(), 0, 1);
    mat.set(angle.sin(), 1, 0);
    mat
}

pub fn shearing(xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Matrix4 {
    let mut mat = Matrix4::ident();
    mat.set(xy, 0, 1);
    mat.set(xz, 0, 2);
    mat.set(yx, 1, 0);
    mat.set(yz, 1, 2);
    mat.set(zx, 2, 0);
    mat.set(zy, 2, 1);
    mat
}

#[cfg(test)]
mod test {
    
#[cfg(test)]
    use crate::coords::{Coord, Point, Vector};
    use std::f64::consts::PI;
    use super::*;
    #[test]
    fn translation_multiplied_by_point_equals_displaced_point() {
        let p = Point::new(5., -3., 2.);
        let transform = translation(-3., 4., 5.);
        let t = &transform*&p;
    
        assert_eq!(t, Point::new(2., 1., 7.));
    }
    
    #[test]
    fn multiplication_by_inverse_of_translation_matrix(){
        let transform = translation(5., -3., 2.);
        let inv = transform.inverse().unwrap();
        let p = Point::new(-3., 4., 5.);
        
        assert_eq!(inv*p, Point::new(-8., 7., 3.));
    }
    
    #[test]
    fn translation_does_not_affect_vectors(){
        let transform = translation(5., -3., 2.);
        let v = Vector::new(-3., 4., 5.);
        
        assert_eq!(transform*v, v);
    }
    
    #[test]
    fn scaling_matrix_on_point(){
        let transform = scaling(2., 3., 4.);
        let p = Point::new(-4., 6., 8.);
        
        assert_eq!(transform*p, Point::new(-8., 18., 32.));
    }
    
    #[test]
    fn scaling_matrix_on_vector(){
        let transform = scaling(2., 3., 4.);
        let p = Vector::new(-4., 6., 8.);
        
        assert_eq!(transform*p, Vector::new(-8., 18., 32.));
    }
    
    #[test]
    fn inv_scaling_matrix_inverses_scaling(){
        let transform = scaling(2., 3., 4.).inverse().unwrap();
        let p = Vector::new(-4., 6., 8.);
        
        assert_eq!(transform*p, Vector::new(-2., 2., 2.));
    }
    
    #[test]
    fn reflection_is_scaling_by_neg_value(){
        let transform = scaling(-1., 1., 1.);
        let p = Point::new(2., 3., 4.);
        
        assert_eq!(transform*p, Point::new(-2., 3., 4.));
    }
    
    #[test]
    fn rotate_point_around_x_axis(){
        let p = Point::new(0., 1., 0.);
        let half_quarter = rotation_x(PI/4.);
        let full_quarter = rotation_x(PI/2.);
    
        assert_eq!(&half_quarter*&p, Point::new(0., f64::sqrt(2.0)/2.0, f64::sqrt(2.0)/2.0));
        assert_eq!(&full_quarter*&p, Point::new(0., 0., 1.0));
    }
    
    #[test]
    fn rotate_point_around_y_axis(){
        let p = Point::new(0., 0., 1.);
        let half_quarter = rotation_y(PI/4.);
        let full_quarter = rotation_y(PI/2.);
    
        assert_eq!(&half_quarter*&p, Point::new(f64::sqrt(2.0)/2.0, 0.0, f64::sqrt(2.0)/2.0));
        assert_eq!(&full_quarter*&p, Point::new(1.0, 0., 0.));
    }
    
    
    #[test]
    fn rotate_point_around_z_axis(){
        let p = Point::new(0., 1., 0.);
        let half_quarter = rotation_z(PI/4.);
        let full_quarter = rotation_z(PI/2.);
    
        assert_eq!(&half_quarter*&p, Point::new(-f64::sqrt(2.0)/2.0,f64::sqrt(2.0)/2.0, 0.0));
        assert_eq!(&full_quarter*&p, Point::new(-1.0, 0., 0.));
    }
    
    #[test]
    fn shearing_transform_moves_axes_proportionately(){
        let transform = shearing(1., 0., 0., 0., 0., 0.);
        let p = Point::new(2., 3., 4.);
    
        assert_eq!(transform*p, Point::new(5., 3., 4.));
    
        let transform = shearing(0., 1., 0., 0., 0., 0.);
        let p = Point::new(2., 3., 4.);
    
        assert_eq!(transform*p, Point::new(6., 3., 4.));
    
        let transform = shearing(0., 0., 1., 0., 0., 0.);
        let p = Point::new(2., 3., 4.);
    
        assert_eq!(transform*p, Point::new(2., 5., 4.));
    
        let transform = shearing(0., 0., 0., 1., 0., 0.);
        let p = Point::new(2., 3., 4.);
    
        assert_eq!(transform*p, Point::new(2., 7., 4.));
    
        let transform = shearing(0., 0., 0., 0., 1., 0.);
        let p = Point::new(2., 3., 4.);
    
        assert_eq!(transform*p, Point::new(2., 3., 6.));
    
        let transform = shearing(0., 0., 0., 0., 0., 1.);
        let p = Point::new(2., 3., 4.);
    
        assert_eq!(transform*p, Point::new(2., 3., 7.));
    }
    
    #[test]
    fn chained_transformations_sequenced_correctly(){
        let p = Point::new(1., 0., 1.);
        let a = rotation_x(PI/2.0);
        let b = scaling(5., 5., 5.);
        let c = translation(10., 5., 7.);
    
        let p2 = a*p;
        assert_eq!(p2, Point::new(1., -1., 0.));
        
        let p3 = b*p2;
        assert_eq!(p3, Point::new(5., -5., 0.));
    
        let p4 = c*p3;
        assert_eq!(p4, Point::new(15., 0., 7.));
    
    }
    
    #[test]
    fn chained_transforms_are_applied_in_reverse_order(){
        let p = Point::new(1., 0., 1.);
        let a = rotation_x(PI/2.0);
        let b = scaling(5., 5., 5.);
        let c = translation(10., 5., 7.);
        let t = c * b * a;
    
        assert_eq!(t*p, Point::new(15., 0., 7.));
    }

}