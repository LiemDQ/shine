use crate::intersections;
use crate::ray::{Ray, Intersection, Material};
use crate::coords::{Point, Coord, Vector};
use crate::matrix::{LinAlg, Matrix4};

/// Sphere object.
/// Currently, the origin is hardcoded to zero, and the radius is hardcoded to 1. 
/// Any changes to the sphere such as translations, rotations and deformations are handled by the
/// transform matrix, which operates on rays that intersect with the sphere. 
/// 
/// This implementation detail may change in the future.
#[derive(Debug, Clone)]
pub struct Sphere {
    origin: Point,
    id: usize,
    radius: f64,
    transform: Matrix4,
    pub material: Material,
}

impl Sphere {
    ///radius is hardcoded for now. Different sphere sizes are handled using a transformation matrix. This may change in the future.
    const RADIUS: f64 = 1.0;

    ///
    pub fn new(id: usize) -> Self {
        
        Self { 
            origin: Point::new(0.,0.,0.), 
            id: id, 
            radius: Sphere::RADIUS, 
            transform: Matrix4::ident() ,
            material: Material::default(),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn set_transform(&mut self, t: Matrix4) {
        self.transform = t;
    }

    pub fn intersect(&self, ray: &Ray) -> Vec<Intersection> {
        let ray2 = ray.transform(&self.transform.inverse().unwrap());
        
        let sphere_to_ray = ray2.origin - self.origin;
        let a = ray2.direction * ray2.direction;
        let b = 2. * ray2.direction * sphere_to_ray;
        let c = sphere_to_ray*sphere_to_ray - self.radius;
        let discriminant = b*b - 4.0*a*c;

        if discriminant < 0.0 {
            return Vec::new();
        }

        let t1 = (-b - discriminant.sqrt() )/(2.0*a);
        let t2 = (-b + discriminant.sqrt() )/(2.0*a);

        intersections![Intersection{t: t1, object: self}, Intersection{t: t2, object: self}]
    }

    pub fn normal_at(&self, pt: &Point) -> Vector {
        //TODO: account for transformation matrix
        let obj_pt = &self.transform.inverse().unwrap() * pt;
        let t = self.transform.inverse().unwrap().transpose();

        let n = t*(&obj_pt - &self.origin);
        n.normalize()
        
    }


}

#[cfg(test)]
mod test {
    use super::*;
    use crate::transforms::{translation, scaling, rotation_z};
    #[test]
    fn change_sphere_transform(){
        let mut s = Sphere::new(1);
        let t = translation(2., 3., 4.);
        s.set_transform(t.clone());
    
        assert_eq!(s.transform, t);
    }
    
    #[test]
    fn normal_on_sphere_at_pt_on_x_axis(){
        let s = Sphere::new(1);
        let n = s.normal_at(&Point::new(1., 0., 0.));
        assert_eq!(n, Vector::new(1., 0., 0.));
    }
    
    #[test]
    fn normal_on_sphere_at_pt_on_y_axis(){
        let s = Sphere::new(1);
        let n = s.normal_at(&Point::new(0., 1., 0.));
        assert_eq!(n, Vector::new(0., 1., 0.));
    }
    
    #[test]
    fn normal_on_sphere_at_pt_on_z_axis(){
        let s = Sphere::new(1);
        let n = s.normal_at(&Point::new(0., 0., 1.));
        assert_eq!(n, Vector::new(0., 0., 1.));
    }
    
    #[test]
    fn normal_on_sphere_at_nonaxial_pt(){
        let s = Sphere::new(1);
        let n = s.normal_at(&Point::new(f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0));
        assert_eq!(n, Vector::new(f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0));
    }
    
    #[test]
    fn normal_is_normalized_vec(){
        let s = Sphere::new(1);
        let n = s.normal_at(&Point::new(f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0, f64::sqrt(3.0)/3.0));
        //normalization is idempotent
        assert_eq!(n, n.normalize());
    }
    
    #[test]
    fn normal_on_translated_sphere(){
        let mut s = Sphere::new(1);
        s.set_transform(translation(0., 1., 0.));
    
        let n = s.normal_at(&Point::new(0., 1.70711, -0.70711));
        assert_eq!(n, Vector::new(0., 0.70711, -0.70711));
    }
    
    #[test]
    fn normal_on_transformed_sphere(){
        let mut s = Sphere::new(1);
        s.set_transform(scaling(1., 0.5, 1.)*rotation_z(std::f64::consts::PI/5.0));
        let n = s.normal_at(&Point::new(0., f64::sqrt(2.0)/2.0, -f64::sqrt(2.0)/2.0));
        assert_eq!(n, Vector::new(0.0, 0.97014, -0.24254));
    }
}
