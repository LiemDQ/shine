
use crate::{canvas::Color, matrix::Matrix4};
use crate::coords::*;
use crate::matrix::Matrix;
use crate::geometry::Sphere;


#[cfg(test)]
use crate::{utils::float_eq, transforms::*};

#[derive(Debug, Clone)]
pub struct Ray {
    pub origin: Point,
    pub direction: Vector,
}

impl Ray {
    pub fn new(origin: Point, direction: Vector) -> Self {
        Self { origin: origin, direction: direction }
    }

    pub fn position(&self, t: f64) -> Point {
        self.origin + self.direction * t
    }

    pub fn transform(&self, m: &Matrix4) -> Self {
        let mut new = self.clone();
        new.origin = m * &new.origin;
        new.direction = m * &new.direction;
        new
    }
}

/// Encapsulates all the information needed for computing intersections
#[derive(Debug,Clone)]
pub struct Intersection<'a> {
    pub t: f64,
    pub object: &'a Sphere, //temporary hack 
}

#[macro_export]
macro_rules! intersections {
    () => {
        Vec<Intersection>::new()
    };

    ($e:expr) => {
        vec![$e]
    };

    ($($x:expr),+ $(,)?) => {{
        use std::cmp::Ordering::Less;
        let mut v = vec![$($x),+];
        v.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Less));
        v
    }};


}

pub fn hit<'a>(intersections: &'a Vec<Intersection<'a>> ) -> Option<&'a Intersection<'a>>{
    for inter in intersections.iter().filter(|i| i.t > 0.0) {
        return Some(inter);
    }
    None
}

pub struct PointLight {
    position: Point,
    intensity: Color,
}

impl PointLight {
    pub fn new(p: Point, c: Color) -> Self {
        Self { position: p, intensity: c }
    }
}

/// Attributes for the Phong reflection model:
/// - Ambient
/// - Diffuse
/// - Specular
/// - Shininess
#[derive(Debug, Clone)]
pub struct Material {
    pub color: Color,
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    pub shininess: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self { 
            color: Color::new(1., 1., 1.), 
            ambient: 0.1, 
            diffuse: 0.9, 
            specular: 0.9, 
            shininess: 200.0 
        }
    }
}

/// Compute a lighting value for a surface from the perspective of an observer (the 'eye') using the Phong reflection model.
/// 
pub fn lighting(material: &Material, pt: &Point, light: &PointLight, eyev: &Vector, normalv: &Vector) -> Color {
    //combine the surface color with the light's color/intensity
    let effective_color = &material.color * &light.intensity;
    
    //find direction to the light source
    let lightv = (&light.position - pt).normalize();
    
    //compute ambient lighting
    let ambient = &effective_color * material.ambient;

    // light dot normal: angle between light vector and normal vector
    //negative value means the light is on the other side of the surface.

    let light_dot_normal = &lightv * normalv;
    let mut diffuse = Color::black();
    let mut specular = Color::black();

    if light_dot_normal > 0.0 {
        //compute diffuse lighting
        diffuse = effective_color*material.diffuse * light_dot_normal;
        
        //reflect_dot_eye represents the cosine of the angle between the reflection vector 
        //and the eye vector. A negative value means the light is being reflected
        //away from the eye.
        let reflectv = -lightv.reflect(normalv);
        let reflect_dot_eye = &reflectv * eyev;
        if reflect_dot_eye > 0.0 {
            let factor = reflect_dot_eye.powf(material.shininess);
            specular = &light.intensity * material.specular * factor;
        }
    }
    ambient + diffuse + specular
}

#[test]
fn create_and_query_ray(){
    let origin = Point::new(1., 2., 3.);
    let direction = Vector::new(4., 5., 6.);
    let ray = Ray::new(origin.clone(), direction.clone());
    
    assert_eq!(ray.origin, origin);
    assert_eq!(ray.direction, direction)
}

#[test]
fn compute_point_from_ray_distance(){
    
    let ray = Ray::new(Point::new(2., 3., 4.), Vector::new(1., 0., 0.));
    
    assert_eq!(ray.position(0.), Point::new(2., 3., 4.));
    assert_eq!(ray.position(1.), Point::new(3., 3., 4.));
    assert_eq!(ray.position(-1.), Point::new(1., 3., 4.));
    assert_eq!(ray.position(2.5), Point::new(4.5, 3., 4.));
}

#[test]
fn ray_intersects_sphere_at_two_points(){
    let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
    let s = Sphere::new(1);
    let xs = s.intersect(&ray);
    assert_eq!(xs.len(), 2);
    assert!(float_eq(xs[0].t, 4.0));
    assert!(float_eq(xs[1].t, 6.0));
}

#[test]
fn ray_originates_from_inside_sphere(){
    let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
    let s = Sphere::new(1);
    let xs = s.intersect(&ray);
    assert_eq!(xs.len(), 2);
    assert!(float_eq(xs[0].t, -1.0));
    assert!(float_eq(xs[1].t, 1.0));

}

#[test]
fn ray_misses_sphere(){
    let ray = Ray::new(Point::new(0., 2., -5.), Vector::new(0., 0., 1.));
    let s = Sphere::new(1);
    let xs = s.intersect(&ray);
    assert_eq!(xs.len(), 0);
}

#[test]
fn ray_intersects_sphere_at_tangent(){
    let ray = Ray::new(Point::new(0., 1., -5.), Vector::new(0., 0., 1.));
    let s = Sphere::new(1);
    let xs = s.intersect(&ray);
    assert_eq!(xs.len(), 2);
    assert_eq!(xs[0].t, xs[1].t);
    assert!(float_eq(xs[0].t, 5.0));
}

#[test]
fn sphere_behind_ray(){
    let ray = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
    let s = Sphere::new(1);
    let xs = s.intersect(&ray);
    assert_eq!(xs.len(), 2);
    assert!(float_eq(xs[0].t, -6.0));
    assert!(float_eq(xs[1].t, -4.0));
}

#[test]
fn intersect_sets_the_object_on_the_intersection(){
    let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
    let s = Sphere::new(1);
    let xs = s.intersect(&ray);
    assert_eq!(xs.len(),2);
    assert_eq!(xs[0].object.id(), 1);
}

#[test]
fn intersection_hit_with_all_positive_t(){
    let s = Sphere::new(1);
    let i1 = Intersection{t: 1., object: &s};
    let i2 = Intersection{t: 2., object: &s};
    let xs = vec![i1.clone(), i2];
    let i = hit(&xs);
    assert_eq!(i.unwrap().t, i1.t);
}

#[test]
fn intersection_hit_with_some_negative_t(){
    let s = Sphere::new(1);
    let i1 = Intersection{t: -1., object: &s};
    let i2 = Intersection{t: 1., object: &s};
    let xs = vec![i1.clone(), i2.clone()];
    let i = hit(&xs);
    assert_eq!(i.unwrap().t, i2.t);
}

#[test]
fn intersection_hit_with_all_negative_t(){
    let s = Sphere::new(1);
    let i1 = Intersection{t: -1., object: &s};
    let i2 = Intersection{t: -2., object: &s};
    let xs = vec![i1, i2];
    let i = hit(&xs);
    
    assert!(i.is_none());
}

#[test]
fn hit_is_lowest_nonnegative_interaction(){
    let s = Sphere::new(1);
    let i1 = Intersection{t: 5., object: &s};
    let i2 = Intersection{t: 7., object: &s};
    let i3 = Intersection{t: -3., object: &s};
    let i4 = Intersection{t: 2., object: &s};
    let xs = intersections![i1, i2, i3, i4.clone()];
    
    let i = hit(&xs).unwrap();
    assert_eq!(i.t, i4.t);
}

#[test]
fn translating_a_ray() {
    let ray = Ray::new(Point::new(1.,2.,3.), Vector::new(0., 1., 0.));
    let m = translation(3., 4., 5.);
    let r2 = ray.transform(&m);
    assert_eq!(r2.origin, Point::new(4., 6., 8.));
    assert_eq!(r2.direction, Vector::new(0., 1., 0.));
}

#[test]
fn scaling_a_ray() {
    let ray = Ray::new(Point::new(1.,2.,3.), Vector::new(0., 1., 0.));
    let m = scaling(2., 3., 4.);
    let r2 = ray.transform(&m);
    assert_eq!(r2.origin, Point::new(2., 6., 12.));
    assert_eq!(r2.direction, Vector::new(0., 3., 0.));
}

#[test]
fn intersect_scaled_sphere_with_ray(){
    let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
    let mut s = Sphere::new(1);
    s.set_transform(scaling(2., 2., 2.));
    let xs = s.intersect(&ray);
    assert_eq!(xs.len(), 2);
    assert!(float_eq(xs[0].t, 3.0));
    assert!(float_eq(xs[1].t, 7.0));
}

#[test]
fn intersect_translated_sphere_with_ray(){
    let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
    let mut s = Sphere::new(1);
    s.set_transform(translation(5., 0., 0.));
    let xs = s.intersect(&ray);
    assert_eq!(xs.len(), 0);
}

#[test]
fn lighting_with_eye_between_light_and_surface(){
    let m = Material::default();
    let p = Point::zero();
    
    //eye is position directly between light and surface. Normal is same as eye.
    let eyev = Vector::new(0., 0., -1.);
    let normalv = Vector::new(0., 0., -1.);
    let light = PointLight::new(Point::new(0., 0., -10.0), Color::white());
    let result = lighting(&m, &p, &light, &eyev, &normalv);
    assert_eq!(result, Color::new(1.9, 1.9, 1.9));
}

#[test]
fn lighting_with_eye_between_light_and_surface_eye_offset_45_deg(){
    let m = Material::default();
    let p = Point::zero();
    
    //eye is position directly between light and surface. Normal is same as eye.
    let eyev = Vector::new(0., f64::sqrt(2.0)/2.0, -f64::sqrt(2.0)/2.0);
    let normalv = Vector::new(0., 0., -1.);
    let light = PointLight::new(Point::new(0., 0., -10.0), Color::white());
    let result = lighting(&m, &p, &light, &eyev, &normalv);
    //specular value should be effectively zero
    assert_eq!(result, Color::new(1.0, 1.0, 1.0));
}

#[test]
fn lighting_with_eye_between_light_and_surface_light_offset_45_deg(){
    let m = Material::default();
    let p = Point::zero();
    
    //eye is position directly between light and surface. Normal is same as eye.
    let eyev = Vector::new(0., 0.0, -1.);
    let normalv = Vector::new(0., 0., -1.);
    let light = PointLight::new(Point::new(0., 10., -10.0), Color::white());
    let result = lighting(&m, &p, &light, &eyev, &normalv);
    //diffuse value is reduced
    //no specular component
    
    assert_eq!(result, Color::new(0.7364, 0.7364, 0.7364));
}

#[test]
fn lighting_with_eye_in_path_of_reflection(){
    let m = Material::default();
    let p = Point::zero();
    
    //eye is position directly between light and surface. Normal is same as eye.
    let eyev = Vector::new(0., -f64::sqrt(2.0)/2.0, -f64::sqrt(2.0)/2.0);
    let normalv = Vector::new(0., 0., -1.);
    let light = PointLight::new(Point::new(0., 10., -10.0), Color::white());
    let result = lighting(&m, &p, &light, &eyev, &normalv);
    
    //specular lighting is at full strength, with diffuse and ambient lighting the same as the previous test
    assert_eq!(result, Color::new(1.6364, 1.6364, 1.6364));
}

#[test]
fn lighting_with_light_behind_surface(){
    let m = Material::default();
    let p = Point::zero();
    
    //eye is position directly between light and surface. Normal is same as eye.
    let eyev = Vector::new(0., 0., -1.);
    let normalv = Vector::new(0., 0., -1.);
    let light = PointLight::new(Point::new(0., 10., 10.0), Color::white());
    let result = lighting(&m, &p, &light, &eyev, &normalv);
    
    //only ambient lighting is left
    assert_eq!(result, Color::new(0.1, 0.1, 0.1));
}