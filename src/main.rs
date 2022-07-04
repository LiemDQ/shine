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
use geometry::{Sphere, Shape, Plane};
use std::{fs, io};
use transforms::*;
use world::*;

use crate::{ray::PointLight, canvas::Color};

fn main() -> io::Result<()> {
    use std::f64::consts::PI;
    let floor = Box::new(Plane::new());


    let mut middle = Box::new(Sphere::new(4));
    middle.set_transform(
        translation(-0.5, 1., 0.5)
    );
    middle.mut_material().color = Color::new(0.1, 1., 0.5);
    middle.mut_material().diffuse = 0.7;
    middle.mut_material().specular = 0.3;

    let mut right = Box::new(Sphere::new(5));
    right.set_transform(
        translation(1.5, 0.5, -0.5)*scaling(0.5, 0.5, 0.5)
    );
    right.mut_material().color = Color::new(0.5, 1.0, 0.1);
    right.mut_material().diffuse = 0.7;
    right.mut_material().specular = 0.3;

    let mut left = Box::new(Sphere::new(6));
    left.set_transform(
        translation(-1.5, 0.33, -0.75)*scaling(0.33, 0.33, 0.33)
    );
    left.mut_material().diffuse = 0.7;
    left.mut_material().specular = 0.3;

    let world = World{
        objects: vec![floor, middle, right, left],
        lights: vec![PointLight::new(Point::new(-10., 10., -10.), Color::new(1., 1., 1.))],
    };

    let mut camera = Camera::new(1000, 500, PI/3.0);
    camera.transform = view_transform(
        Point::new(0., 1.5, -5.), 
        Point::new(0., 1., 0.),
        Vector::new(0., 1., 0.)
    );

    let canvas = camera.render(&world);

    fs::write("sphere_floor.ppm", canvas.to_ppm())?;
    Ok(())
}
