use crate::{intersections, world};
use crate::ray::{Ray, Intersection, Material};
use crate::coords::{Point, Coord, Vector};
use crate::matrix::{LinAlg, Matrix4};
use std::fmt::Debug;

pub trait Shape: Debug {
    fn set_transform(&mut self, t: Matrix4);
    fn set_material(&mut self, m: Material);
    fn transform(&self) -> Matrix4;
    fn material(&self) -> &Material;
    fn mut_material(&mut self) -> &mut Material;
    fn local_intersect(&self, ray: &Ray) -> Vec<Intersection>;
    fn local_normal_at(&self, pt: &Point) -> Vector;

    fn normal_at(&self, pt: &Point) -> Vector {
        let local_pt = &self.transform().inverse().unwrap() * pt;
        let local_normal = self.local_normal_at(&local_pt);
        let world_normal = self.transform().inverse().unwrap().transpose() * local_normal;
        world_normal.normalize()
    }

    fn intersect(&self, ray: &Ray) -> Vec<Intersection> {
        let local_ray = ray.transform(&self.transform().inverse().unwrap());
        self.local_intersect(&local_ray)    
    }
}

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
    material: Material,
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
}

impl Shape for Sphere {
    fn set_transform(&mut self, t: Matrix4) {
        self.transform = t;
    }

    fn transform(&self) -> Matrix4 {
        self.transform
    }

    fn set_material(&mut self, m: Material) {
        self.material = m;
    }

    fn material(&self) -> &Material {
        &self.material
    }

    fn mut_material(&mut self) -> &mut Material {
        &mut self.material
    }

    fn local_intersect(&self, ray: &Ray) -> Vec<Intersection> {
        
        let sphere_to_ray = ray.origin - self.origin;
        let a = ray.direction * ray.direction;
        let b = 2. * ray.direction * sphere_to_ray;
        let c = sphere_to_ray*sphere_to_ray - self.radius;
        let discriminant = b*b - 4.0*a*c;

        if discriminant < 0.0 {
            return Vec::new();
        }

        let t1 = (-b - discriminant.sqrt() )/(2.0*a);
        let t2 = (-b + discriminant.sqrt() )/(2.0*a);

        intersections![Intersection{t: t1, object: self}, Intersection{t: t2, object: self}]
    }

    fn local_normal_at(&self, pt: &Point) -> Vector {
        pt - &self.origin
    }
}

/// A plane is a flat surface that extends infinitely in two dimensions.
#[derive(Debug, Clone)]
pub struct Plane {
    transform: Matrix4,
    material: Material,
}

impl Plane {
    pub fn new() -> Self {
        Self { 
            transform: Matrix4::ident(), 
            material: Material::default(), 
            }
    }

}

impl Shape for Plane {
    fn material(&self) -> &Material {
        &self.material
    }

    fn mut_material(&mut self) -> &mut Material {
        &mut self.material
    }

    fn set_material(&mut self, m: Material) {
        self.material = m;
    }

    fn set_transform(&mut self, t: Matrix4) {
        self.transform = t;
    }

    fn transform(&self) -> Matrix4 {
        self.transform
    }

    fn local_intersect(&self, ray: &Ray) -> Vec<Intersection> {
        //case 1: ray is parallel to the plane
        //case 2: ray is coplanar to the plane
        //this is true if the ray has no y-component.
        if ray.direction.y.abs() < 0.00001 {
            return Vec::new();
        }
        //case 3: ray origin is above the plane
        //case 4: ray origin is below the plane
        let t = -ray.origin.y/ray.direction.y;
        
        vec![Intersection{t, object: self}]
    }

    fn local_normal_at(&self, pt: &Point) -> Vector {
        Vector { x: 0.0, y: 1.0, z: 0.0 }        
    }

}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{transforms::{translation, scaling, rotation_z}, utils::float_eq};
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

    #[test]
    fn normal_on_plane_is_always_the_same(){
        let plane = Plane::new();
        let vec = Vector::new(0.0, 1.0, 0.0);
        let n1 = plane.local_normal_at(&Point::new(0.0, 0.0, 0.0));
        let n2 = plane.local_normal_at(&Point::new(10.0, 0.0, -10.0));
        let n3 = plane.local_normal_at(&Point::new(-5.0, 0.0, 150.0));

        assert_eq!(vec, n1);
        assert_eq!(vec, n2);
        assert_eq!(vec, n3);
    }

    #[test]
    fn intersect_with_ray_parallel_to_plane(){
        let plane = Plane::new();
        let r = Ray::new(Point::new(0.0, 10.0, 0.0), Vector::new(0.0, 0.0, 1.0));
        let xs = plane.local_intersect(&r);
        assert!(xs.is_empty());
    }

    #[test]
    fn intersect_plane_with_coplanar_ray(){
        let plane = Plane::new();
        let r = Ray::new(Point::new(0.0, 00.0, 0.0), Vector::new(0.0, 0.0, 1.0));
        let xs = plane.local_intersect(&r);
        assert!(xs.is_empty());
    }

    #[test]
    fn intersect_plane_with_ray_from_above(){
        let plane = Plane::new();
        let r = Ray::new(Point::new(0.0, 1.0, 0.0), Vector::new(0.0, -1.0, 0.0));
        let xs = plane.local_intersect(&r);
        assert_eq!(xs.len(), 1);
        assert!(float_eq(xs[0].t, 1.0));
    }

    #[test]
    fn intersect_plane_with_ray_from_below(){
        let plane = Plane::new();
        let r = Ray::new(Point::new(0.0, -1.0, 0.0), Vector::new(0.0, 1.0, 0.0));
        let xs = plane.local_intersect(&r);
        assert_eq!(xs.len(), 1);
        assert!(float_eq(xs[0].t, 1.0));
    }

    
}
