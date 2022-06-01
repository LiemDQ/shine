#[macro_use]
mod utils;
mod coords;
mod canvas;
mod matrix;
mod transforms;
mod ray;
mod geometry;
mod world;

use coords::{Point, Vector, Coord};
use canvas::{Canvas, Pixel};
use geometry::Sphere;
use std::{fs, io};
use matrix::{Matrix, Matrix4, LinAlg};
use transforms::*;

use ray::{Ray, Intersection, hit, lighting};

use crate::{ray::PointLight, canvas::Color};

const CANVAS_DIM: usize = 1000;

struct Projectile {
    pub position: Point,
    pub velocity: Vector,
}

struct Environment {
    pub gravity: Vector,
    pub wind: Vector,
}

fn tick(envir: &Environment, proj: &mut Projectile){
    proj.position = proj.position + proj.velocity;
    proj.velocity = proj.velocity + envir.gravity + envir.wind;
}

fn imprint_proj_on_canvas(canvas: &mut Canvas, proj: &Projectile) {
    const PX: Pixel = Pixel::new(1.0, 0., 0.);
    canvas.write_pixel(proj.position.x as usize, canvas.height() - proj.position.y as usize - 1, PX);
}

fn draw_point_on_canvas(canvas: &mut Canvas, pt: &Point) {
    const PX: Pixel = Pixel::new(1.0, 1.0, 1.0);
    canvas.write_pixel((pt.x as i64 + CANVAS_DIM as i64/2 ) as usize, (pt.y as i64 + CANVAS_DIM as i64/2 ) as usize, PX);
}


fn main() -> io::Result<()> {

    const RED: Pixel = Pixel::new(1.0, 0., 0.);
    const PINK: Pixel = Pixel::new(1.0, 0.2, 1.);

    let mut canvas = Canvas::new(CANVAS_DIM,CANVAS_DIM);
    let origin = Point::new(0., 0., -5.);
    let wall_z = 10.0;
    let wall_size = 7.0;
    let pixel_size = wall_size/(CANVAS_DIM as f64);
    let half = wall_size / 2.0;
    let mut shape = Sphere::new(1);
    shape.material.color = PINK;

    //shape.set_transform(scaling(1., 0.5, 1.));

    //add a light source.
    let light = PointLight::new(Point::new(0., -10., 0.), Color::white());

    for y in 0..CANVAS_DIM {
        //y coordinate of world, with the top = +half and bottom = -half
        let world_y = half - pixel_size*(y as f64);
        
        for x in 0..CANVAS_DIM {
            //x coordinate of world, from left to right
            let world_x = -half + pixel_size*(x as f64);
            let target = Point::new(world_x, world_y, wall_z);
            let r = Ray::new(origin.clone(), (target-origin).normalize());
            let xs = shape.intersect(&r);
            if let Some(inter) = hit(&xs) {
                let point = r.position(inter.t);
                let normal = inter.object.normal_at(&point);
                let eye = -r.direction;
                let color = lighting(&inter.object.material,  &point, &light, &eye, &normal);
                canvas.write_pixel(x, y, color);
            }
        }

    }

    fs::write("sphere_lit.ppm", canvas.to_ppm())?;
    Ok(())
}
