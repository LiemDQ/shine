
use std::fmt::Debug;

use crate::matrix::{LinAlg};
use crate::{canvas::Color, matrix::Matrix4};
use crate::coords::*;
use crate::geometry::{Sphere, Shape};


#[cfg(test)]
use crate::{utils::float_eq, transforms::*};

#[derive(Debug, Clone, Copy)]
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
        let mut new = *self;
        new.origin = m * &new.origin;
        new.direction = m * &new.direction;
        new
    }
}

/// Encapsulates all the information needed for computing intersections
#[derive(Debug,Clone)]
pub struct Intersection<'a> {
    pub t: f64,
    pub object: &'a dyn Shape,  
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
    pub position: Point,
    pub intensity: Color,
}

impl PointLight {
    pub fn new(p: Point, c: Color) -> Self {
        Self { position: p, intensity: c }
    }
}

pub trait Pattern: Debug {
    fn at(&self, pt: &Point) -> Color;
    fn transform(&self) -> Matrix4;
    fn set_transform(&mut self, t: Matrix4);
    fn at_object(&self, object: &dyn Shape, pt: &Point) -> Color {
        let object_pt = &object.transform().inverse().unwrap() * pt;
        let pattern_pt = self.transform().inverse().unwrap()* object_pt;
        self.at(&pattern_pt)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SolidPattern {
    a: Color,
}

impl SolidPattern {
    pub fn new(a: Color) -> Box<Self> {
        Box::new(Self {a})
    }
}

impl Pattern for SolidPattern {
    fn at(&self, _: &Point) -> Color {
        self.a
    }

    fn transform(&self) -> Matrix4 {
        Matrix4::ident()
    }

    fn set_transform(&mut self, _: Matrix4) {
        //noop
    }

}

#[derive(Debug, Copy, Clone)]
pub struct StripePattern {
    a: Color,
    b: Color,
    transform: Matrix4
}


impl StripePattern {
    pub fn new(a: Color, b: Color) -> Box<Self> { 
        Box::new(Self { a, b, transform: Matrix4::ident() })
    }

    fn color(&self, pt: &Point) -> Color {
        if pt.x.floor() as i64 % 2 == 0 {
            self.a
        } else {
            self.b
        }
    }
}

impl Pattern for StripePattern {
    fn at(&self, pt: &Point) -> Color {
        self.color(pt)
    }

    fn transform(&self) -> Matrix4 {
        self.transform
    }

    fn set_transform(&mut self, t: Matrix4) {
        self.transform = t;
    }
}

#[derive(Debug, Copy, Clone)]
pub struct GradientPattern {
    a: Color,
    b: Color,
    transform: Matrix4
}

impl GradientPattern {
    pub fn new(a: Color, b: Color) -> Box<Self> {
        Box::new(Self { a, b, transform: Matrix4::ident() })
    }
}

impl Pattern for GradientPattern {
    fn set_transform(&mut self, t: Matrix4) {
        self.transform = t;
    }
    
    fn transform(&self) -> Matrix4 {
        self.transform
    }

    fn at(&self, pt: &Point) -> Color {
        self.a + (self.b - self.a) * (pt.x - pt.x.floor())
    }
} 

#[derive(Debug, Copy, Clone)]
pub struct RingPattern {
    a: Color,
    b: Color,
    transform: Matrix4
}

impl RingPattern {
    pub fn new(a: Color, b: Color) -> Box<Self> {
        Box::new(Self { a, b, transform: Matrix4::ident() })
    }
}

impl Pattern for RingPattern {
    fn set_transform(&mut self, t: Matrix4) {
        self.transform = t;
    }
    
    fn transform(&self) -> Matrix4 {
        self.transform
    }

    fn at(&self, pt: &Point) -> Color {
        if f64::sqrt(pt.x*pt.x + pt.z*pt.z).floor() as i64 % 2 == 0 {
            self.a
        } else {
            self.b
        }
    }
} 


#[derive(Debug, Copy, Clone)]
pub struct CheckerPattern {
    a: Color,
    b: Color,
    transform: Matrix4
}

impl CheckerPattern {
    pub fn new(a: Color, b: Color) -> Box<Self> {
        Box::new(Self { a, b, transform: Matrix4::ident() })
    }
}

impl Pattern for CheckerPattern {
    fn set_transform(&mut self, t: Matrix4) {
        self.transform = t;
    }
    
    fn transform(&self) -> Matrix4 {
        self.transform
    }

    fn at(&self, pt: &Point) -> Color {
        if (pt.x.floor() + pt.y.floor() + pt.z.floor()) as i64 % 2 == 0 {
            self.a
        } else {
            self.b
        }
    }
} 



/// Attributes for the Phong reflection model:
/// - Ambient
/// - Diffuse
/// - Specular
/// - Shininess
#[derive(Debug)]
pub struct Material {
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    pub shininess: f64,
    pub pattern: Box<dyn Pattern>,
}

impl Default for Material {
    fn default() -> Self {
        Self { 
            ambient: 0.1, 
            diffuse: 0.9, 
            specular: 0.9, 
            shininess: 200.0 ,
            pattern: SolidPattern::new(Color::new(1., 1., 1.))
        }
    }
}

impl Material {
    pub fn new(ambient: f64, diffuse: f64, specular: f64, shininess: f64, pattern: Box<dyn Pattern> ) -> Self {
        Self { ambient, diffuse, specular, shininess, pattern }
    }

    pub fn color(&self, object: &dyn Shape, pt: &Point) -> Color {
        self.pattern.at_object(object, pt)
    }
}

/// Compute a lighting value for a surface from the perspective of an observer (the 'eye') 
/// using the Phong reflection model.
pub fn lighting(
    material: &Material,  //object material
    object: &dyn Shape,
    pt: &Point, //point on the object that the light hits
    light: &PointLight, //light used to shine
    eyev: &Vector, //position of eye
    normalv: &Vector, //normal vector of surface of object
    in_shadow: bool //whether or not the surface is occluded 
) -> Color { 
    //combine the surface color with the light's color/intensity
    let effective_color = &material.color(object, pt) * &light.intensity;
    
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
            //approximation of specular lighting
            let factor = reflect_dot_eye.powf(material.shininess);
            specular = &light.intensity * material.specular * factor;
        }
    }
    if in_shadow {
        ambient
    } else {
        ambient + diffuse + specular
    }
}

pub struct Computations<'a> {
    pub t: f64,
    pub object: &'a dyn Shape,
    pub point: Point,
    pub over_point: Point,
    pub eyev: Vector,
    pub normalv: Vector,
    pub inside: bool,
}

pub fn prepare_computations<'a>(intersection: &'a Intersection, ray: &Ray) -> Computations<'a> {
    let point = ray.position(intersection.t);
    let mut normalv = intersection.object.normal_at(&point);
    let inside = if normalv * -ray.direction < 0.0 {
        normalv = -normalv;
        true
    } else {
        false
    };
    let over_point = point + normalv * 0.00001;
    Computations { 
        t: intersection.t, 
        object: intersection.object, 
        point: point, 
        over_point: over_point,
        eyev: -ray.direction, 
        normalv: normalv,
        inside: inside
    }
}

#[cfg(test)]
mod test {
    use super::*;
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
        // assert_eq!(xs[0].object.id(), 1);
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
        let sphere = Sphere::new(0);
        let result = lighting(&m, &sphere, &p, &light, &eyev, &normalv, false);
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
        let sphere = Sphere::new(0);
        let result = lighting(&m, &sphere, &p, &light, &eyev, &normalv, false);
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
        let sphere = Sphere::new(0);
        let result = lighting(&m, &sphere, &p, &light, &eyev, &normalv, false);
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
        let sphere = Sphere::new(0);
        let result = lighting(&m, &sphere, &p, &light, &eyev, &normalv, false);
        
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
        let sphere = Sphere::new(0);
        let result = lighting(&m, &sphere, &p, &light, &eyev, &normalv, false);
        
        //only ambient lighting is left
        assert_eq!(result, Color::new(0.1, 0.1, 0.1));
    }
    
    #[test]
    fn lighting_with_the_surface_in_shadow(){
        let m = Material::default();
        let pt = Point::zero();
        let eyev = Vector::new(0., 0., -1.);
        let normalv = Vector::new(0., 0., -1.);
        let light = PointLight::new(Point::new(0., 0., -10.), Color::new(1., 1., 1.));
        let in_shadow = true;
        let sphere = Sphere::new(0);
        let result = lighting(&m, &sphere, &pt, &light, &eyev, &normalv, in_shadow);
        assert_eq!(result, Color::new(0.1, 0.1, 0.1));
    }
    
    #[test]
    fn precompute_state_of_intersection(){
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let shape = Sphere::new(1);
        let i = Intersection {t: 4., object: &shape};
        let comps = prepare_computations(&i, &r);
        assert_eq!(comps.t, i.t);
        assert_eq!(comps.point, Point::new(0., 0., -1.));
        assert_eq!(comps.normalv, Vector::new(0., 0., -1.));
    }
    
    #[test]
    fn precompute_state_of_intersection_outside(){
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let shape = Sphere::new(1);
        let i = Intersection {t: 4., object: &shape};
        let comps = prepare_computations(&i, &r);
        assert_eq!(comps.t, i.t);
        assert_eq!(comps.point, Point::new(0., 0., -1.));
        assert_eq!(comps.normalv, Vector::new(0., 0., -1.));
        assert_eq!(comps.inside, false);
    }
    
    #[test]
    fn precompute_state_of_intersection_inside(){
        let r = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let shape = Sphere::new(1);
        let i = Intersection {t: 1., object: &shape};
        let comps = prepare_computations(&i, &r);
        assert_eq!(comps.t, i.t);
        assert_eq!(comps.point, Point::new(0., 0., 1.));
        assert_eq!(comps.eyev, Vector::new(0., 0., -1.));
        assert_eq!(comps.normalv, Vector::new(0., 0., -1.));
        assert_eq!(comps.inside, true)
    }
    
    #[test]
    fn hit_should_offset_point(){
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let mut shape = Sphere::new(1);
        shape.set_transform(translation(0., 0., 1.));
        let i = Intersection {t: 5., object: &shape};
        let comps = prepare_computations(&i, &r);
        assert!(comps.over_point.z < -0.00001/2.);
        assert!(comps.point.z > comps.over_point.z );
    }

    use crate::coords::Coord;

    use super::*;
    #[test]
    fn stripe_pattern_constant_in_y(){
        let pattern = StripePattern::new(Color::white(), Color::black());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 0.0)), Color::white());
        assert_eq!(pattern.at(&Point::new(0.0, 1.0, 0.0)), Color::white());
        assert_eq!(pattern.at(&Point::new(0.0, 2.0, 0.0)), Color::white());
    }

    #[test]
    fn stripe_pattern_constant_in_z(){
        let pattern = StripePattern::new(Color::white(), Color::black());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 0.0)), Color::white());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 1.0)), Color::white());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 2.0)), Color::white());
    }

    
    #[test]
    fn stripe_pattern_alternates_in_x(){
        let pattern = StripePattern::new(Color::white(), Color::black());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 0.0)), Color::white());
        assert_eq!(pattern.at(&Point::new(1.0, 0.0, 0.0)), Color::black());
        assert_eq!(pattern.at(&Point::new(2.0, 0.0, 0.0)), Color::white());
    }

    #[test]
    fn lighting_with_pattern_applied(){
        
        let m = Material {
            ambient: 1.0,
            diffuse: 0.0,
            specular: 0.0,
            shininess: 200.0,
            pattern: StripePattern::new(Color::white(), Color::black())
        };
        let eyev = Vector::new(0.0, 0.0, -1.0);
        let normalv = Vector::new(0.0, 0.0, -1.0);
        let light = PointLight::new(Point::new(0.0, 0.0, -10.0), Color::new(1.0, 1.0, 1.0));
        let sphere = Sphere::new(0);
        let c1 = lighting(&m, &sphere, &Point::new(0.9, 0.0, 0.0), &light, &eyev,  &normalv, false);
        let c2 = lighting(&m, &sphere, &Point::new(1.1, 0.0, 0.0), &light, &eyev,  &normalv, false);

        assert_eq!(c1, Color::white());
        assert_eq!(c2, Color::black());

    }

    #[test]
    fn stripes_with_object_transformation(){
        let mut object = Sphere::new(1);
       object.set_transform(scaling(2., 2., 2.));
       let pattern = StripePattern::new(Color::white(), Color::black());

       let c= pattern.at_object(&object, &Point::new(1.5, 0.0, 0.0));
       assert_eq!(c, Color::white());
    }

    #[test]
    fn stripes_with_pattern_transformation(){
        let mut object = Sphere::new(1);
        object.set_transform(scaling(2., 2., 2.));
        let pattern = StripePattern::new(Color::white(), Color::black());
        let c= pattern.at_object(&object, &Point::new(1.5, 0.0, 0.0));
        assert_eq!(c, Color::white());
    }

    #[test]
    fn stripes_with_pattern_and_object_transformation(){
        let mut object = Sphere::new(1);
        object.set_transform(translation(0.5, 0., 0.));
        let pattern = StripePattern::new(Color::white(), Color::black());
        let c= pattern.at_object(&object, &Point::new(2.5, 0.0, 0.0));
        assert_eq!(c, Color::white());
    }

    #[test]
    fn gradient_pattern_linear_blend(){
        let pattern = GradientPattern::new(Color::white(), Color::black());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 0.0)), Color::white());
        assert_eq!(pattern.at(&Point::new(0.25, 0.0, 0.0)), Color::new(0.75, 0.75, 0.75));
        assert_eq!(pattern.at(&Point::new(0.5, 0.0, 0.0)), Color::new(0.5, 0.5, 0.5));
        assert_eq!(pattern.at(&Point::new(0.75, 0.0, 0.0)), Color::new(0.25, 0.25, 0.25));
    }

    #[test]
    fn ring_pattern_extension_in_x_and_z(){
        let pattern = RingPattern::new(Color::white(), Color::black());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 0.0)), Color::white());
        assert_eq!(pattern.at(&Point::new(1.0, 0.0, 0.0)), Color::black());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 1.0)), Color::black());
        assert_eq!(pattern.at(&Point::new(0.708, 0.0, 0.708)), Color::black());
    }

    #[test]
    fn checker_pattern_repeating() {
        let pattern = CheckerPattern::new(Color::white(), Color::black());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 0.0)), Color::white());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 0.99)), Color::white());
        assert_eq!(pattern.at(&Point::new(0.0, 0.0, 1.01)), Color::black());
    }
}
