use crate::canvas::{Color, Canvas};
use crate::geometry::Sphere;
use crate::matrix::{Matrix4, LinAlg};
use crate::ray::{PointLight, Material, Ray, Intersection, Computations, prepare_computations, lighting};
use crate::coords::{Point, Coord, Vector, cross};
use crate::transforms::{scaling, translation};
use std::cmp::Ordering::Less;

#[cfg(test)]
use crate::utils::float_approx;

pub struct World {
    pub objects: Vec<Sphere>,
    pub lights: Vec<PointLight>,
}

impl Default for World {
    fn default() -> Self {
        let light = PointLight::new(Point::new(-10., 10., -10.), Color::new(1., 1., 1.));
        let mut s1 = Sphere::new(0);
        s1.material = Material{
            color: Color::new(0.8, 1.0, 0.6),
            specular: 0.2,
            diffuse: 0.7,
            shininess: 200.0,
            ambient: 0.1,
        };
        let mut s2 = Sphere::new(1);
        s2.set_transform(scaling(0.5, 0.5, 0.5));
        Self { objects: vec![s1, s2], lights: vec![light] }
    }
}

impl World {
    pub fn intersect(&self, ray: &Ray) -> Vec<Intersection> {
        let mut result = self.objects
            .iter()
            .flat_map(|obj| obj.intersect(ray))
            .filter(|int| int.t > 0.0)
            .collect::<Vec<Intersection>>();
        result.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Less));
        result
    }

    pub fn shade_hit(&self, comps: &Computations) -> Color {
        lighting(
            &comps.object.material, 
            &comps.point, 
            self.lights.first().unwrap(), 
            &comps.eyev, 
            &comps.normalv,
            self.is_shadowed(&comps.over_point)
        )
    }

    pub fn color_at(&self, ray: &Ray) -> Color {
        let intersections = self.intersect(ray);
        //the hit is the closest intersection to the light source. 
        //This is the intersection with the smallest t value.
        let hit = if let Some(hit) = intersections.first() {
            hit
        } else {
            return Color::black();
        };
        let comps = prepare_computations(hit, ray);
        self.shade_hit(&comps)
    }

    fn is_shadowed(&self, pt: &Point) -> bool {
        let shadow_vec =  &self.lights[0].position - pt;
        let dist = shadow_vec.magnitude();
        let direction = shadow_vec.normalize();
        let shadow_ray = Ray::new(*pt, direction);
        let intersects = self.intersect(&shadow_ray);
        if !intersects.is_empty() && intersects[0].t < dist {
            true
        } else {
            false
        }
    }
}

pub fn view_transform(from: Point, to: Point, up: Vector) -> Matrix4 {
    let forward = (to - from).normalize();
    let left = cross(forward, up.normalize());
    let true_up = cross(left, forward);
    let orientation_data = [
        [left.x, left.y, left.z, 0.],
        [true_up.x, true_up.y, true_up.z, 0.],
        [-forward.x, -forward.y, -forward.z, 0.],
        [0., 0., 0., 1.],
    ];
    let orientation = Matrix4::construct(orientation_data);
    orientation*translation(-from.x, -from.y, -from.z)
}

pub struct Camera {
    pub hsize: usize,
    pub vsize: usize, 
    pub field_of_view: f64,
    pub transform: Matrix4,
    pixel_size: f64,
    half_width: f64,
    half_height: f64,
}

impl Camera {
    pub fn new(hsize: usize, vsize: usize, field_of_view: f64) -> Self {
        let aspect_ratio = hsize as f64 / vsize as f64; 
        let half_view= (field_of_view/2.0).tan();

        let (half_width, half_height ) = if aspect_ratio >= 1. {
            (half_view, half_view/aspect_ratio)
        } else {
            (half_view*aspect_ratio, half_view)
        };
        
        let pixel_size = (half_width*2.0)/hsize as f64;
        Self { hsize, vsize, field_of_view, transform: Matrix4::ident(), pixel_size, half_width, half_height }
    }
    
    ///Returns a ray that starts at the camera and passes through the indicated `x, y` pixel
    ///on the canvas. 
    pub fn ray_for_pixel(&self, px: usize, py: usize) -> Ray {
        let px = px as f64;
        let py = py as f64;
        //the offset from the edge of the canvas to the pixel's center
        let xoffset = (px+0.5)*self.pixel_size;
        let yoffset = (py+0.5)*self.pixel_size;
        
        //the untransformed coordinates of the pixel in world space
        //the camera looks towards -z, so +x is to the left
        let world_x = self.half_width - xoffset;
        let world_y = self.half_height - yoffset;

        let inv = self.transform.inverse().unwrap();

        let pixel = inv*Point::new(world_x, world_y, -1.);
        let origin = inv*Point::zero();
        let direction = (pixel - origin).normalize();
        Ray::new(origin, direction)
    }

    pub fn render(&self, world: &World) -> Canvas {
        let mut canvas = Canvas::new(self.hsize, self.vsize);
        for y in 0..self.vsize-1 {
            for x in 0..self.hsize-1 {
                let ray = self.ray_for_pixel(x, y);
                let color = world.color_at(&ray);
                canvas.write_pixel(x, y, color);
            }
        }
        canvas
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn intersect_world_with_ray() {
        let world = World::default();
        let r = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let xs = world.intersect(&r);
        assert_eq!(xs.len(), 4);
        assert_eq!(xs[0].t, 4.0);
        assert_eq!(xs[1].t, 4.5);
        assert_eq!(xs[2].t, 5.5);
        assert_eq!(xs[3].t, 6.0);
    }
    
    #[test]
    fn shade_world_intersection(){
        let ray = Ray::new(Point::new(0., 0., -5.), Vector::new(0., 0., 1.));
        let world = World::default();
        let shape = world.objects.first().unwrap();
        let i = Intersection {t: 4., object: shape};
        let comps = prepare_computations(&i, &ray);
        let c = world.shade_hit(&comps);
    
        assert_eq!(c, Color::new(0.38066, 0.47583, 0.2855));
    }
    
    
    #[test]
    fn shade_world_intersection_inside(){
        let mut world = World::default();
        world.lights[0] = PointLight::new(Point::new(0., 0.25, 0.), Color::new(1., 1., 1.));
        let ray = Ray::new(Point::new(0., 0., 0.), Vector::new(0., 0., 1.));
        let shape = &world.objects[1];
        let i = Intersection {t: 0.5, object: shape};
        let comps = prepare_computations(&i, &ray);
        let c = world.shade_hit(&comps);
    
        assert_eq!(c, Color::new(0.90498, 0.90498, 0.90498));
    }
    
    #[test]
    fn shade_hit_given_intersection_in_shadow(){
        let mut w= World::default();
        w.lights[0] = PointLight::new(Point::new(0., 0., -10.), Color::new(1., 1., 1.));
        let s1 = Sphere::new(1);
        let mut s2 = Sphere::new(2);
        s2.set_transform(translation(0., 0., 10.));
        let r = Ray::new(Point::new(0., 0., 5.), Vector::new(0., 0., 1.));
        w.objects = vec![s1, s2];
        let i = Intersection {t: 4., object: &w.objects[1]};
        let comps = prepare_computations(&i, &r);
        let c = w.shade_hit(&comps);
        assert_eq!(c, Color::new(0.1, 0.1, 0.1))
    }
    
    #[test]
    fn color_with_intersection_behind_ray(){
        let mut w = World::default();
        let outer = w.objects.first_mut().unwrap();
        outer.material.ambient = 1.0; 
        let inner = &mut w.objects[1];
        inner.material.ambient = 1.0;
        let inner_color = inner.material.color;
    
        let ray = Ray::new(Point::new(0., 0., 0.75), Vector::new(0., 0., -1.));
        let c = w.color_at(&ray);
        assert_eq!(c, inner_color);
    }
    
    #[test]
    fn view_transformation_matrix_for_default_orientation(){
        let from = Point::new(0.,0.,0.);
        let to = Point::new(0.,0.,-1.);
        let up = Vector::new(0.,1.,0.);
        let t = view_transform(from, to, up);
        assert_eq!(t, Matrix4::ident());
    }
    
    #[test]
    fn view_transformation_matrix_for_positive_z_orientation(){
        let from = Point::new(0.,0.,0.);
        let to = Point::new(0.,0.,1.);
        let up = Vector::new(0.,1.,0.);
        let t = view_transform(from, to, up);
        assert_eq!(t, scaling(-1., 1., -1.));
    }
    
    #[test]
    fn view_transformation_moves_world(){
        let from = Point::new(0.,0.,8.);
        let to = Point::new(0.,0.,0.);
        let up = Vector::new(0.,1.,0.);
        let t = view_transform(from, to, up);
        assert_eq!(t, translation(0., 0., -8.));
    }
    
    #[test]
    fn view_transformation_matrix_for_arbitrary_orientation(){
        let from = Point::new(1.,3.,2.);
        let to = Point::new(4.,-2.,8.);
        let up = Vector::new(1.,1.,0.);
        let t = view_transform(from, to, up);
        let ans = [
            [ -0.50709, 0.50709, 0.67612, -2.36643],
            [ 0.76772, 0.60609, 0.12122, -2.82843],
            [ -0.35857, 0.59761, -0.71714, 0.00000],
            [ 0.00000, 0.00000, 0.00000, 1.00000],
        ];
    
        let t2 = Matrix4::construct(ans);
    
        assert_eq!(t, t2);
    }
    
    
    
    #[cfg(test)]
    use std::f64::consts::PI;
    
    #[test]
    fn pixel_size_for_horizontal_canvas() {
        let c = Camera::new(200, 125, PI/2.0);
        assert!(float_approx(c.pixel_size, 0.01, 0.00001));
    }
    
    #[test]
    fn pixel_size_for_vertical_canvas(){
        let c = Camera::new(125, 200, PI/2.0);
        assert!(float_approx(c.pixel_size, 0.01, 0.00001));
    }
    
    #[test]
    fn ray_through_center_of_canvas(){
        let c= Camera::new(201, 101, PI/2.0);
        let r= c.ray_for_pixel(100, 50);
        assert_eq!(r.origin, Point::zero());
        assert_eq!(r.direction, Vector::new(0., 0., -1.));
    }
    
    #[test]
    fn ray_through_corner_of_canvas(){
        let c= Camera::new(201, 101, PI/2.0);
        let r= c.ray_for_pixel(0, 0);
        assert_eq!(r.origin, Point::zero());
        assert_eq!(r.direction, Vector::new(0.66519, 0.33259, -0.66851));
    }
    
    #[test]
    fn ray_when_camera_is_transformed(){
        use crate::transforms::rotation_y;
        let mut c= Camera::new(201, 101, PI/2.0);
        c.transform = rotation_y(PI/4.0)*translation(0., -2., 5.);
        let r= c.ray_for_pixel(100, 50);
        assert_eq!(r.origin, Point::new(0., 2., -5.));
        assert_eq!(r.direction, Vector::new(f64::sqrt(2.0)/2.0, 0., -f64::sqrt(2.0)/2.0));
    }
    
    #[test]
    fn render_world_with_camera(){
        let w = World::default();
        let mut c = Camera::new(11, 11, PI/2.0);
        let from = Point::new(0., 0., -5.);
        let to = Point::new(0., 0., 0.);
        let up = Vector::new(0., 1., 0.);
        c.transform = view_transform(from, to, up);
        let image = c.render(&w);
        assert_eq!(image.pixel_at(5, 5), &Color::new(0.38066, 0.47583, 0.2855));
    }
    
    #[test]
    fn no_shadow_when_nothing_is_colinear_with_point_and_light(){
        let w = World::default();
        let p = Point::new(0., 10., 0.);
        assert!(!w.is_shadowed(&p));
    }
    
    
    #[test]
    fn shadowed_when_object_between_point_and_light(){
        let w = World::default();
        let p = Point::new(10., -10., 10.);
        assert!(w.is_shadowed(&p));
    }
    
    #[test]
    fn shadow_when_an_object_is_behind_the_light(){
        let w = World::default();
        let p = Point::new(-20., 20., -20.);
        assert!(!w.is_shadowed(&p));
    }
    
    #[test]
    fn no_shadow_when_object_behind_point(){
        let w = World::default();
        let p = Point::new(-2., 2., -2.);
        assert!(!w.is_shadowed(&p));
    }
}