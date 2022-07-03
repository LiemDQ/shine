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
use geometry::Sphere;
use std::{fs, io};
use transforms::*;
use world::*;

use crate::{ray::PointLight, canvas::Color};

fn main() -> io::Result<()> {
    use std::f64::consts::PI;
    let mut floor = Sphere::new(1);
    floor.set_transform(scaling(10., 0.01, 10.));
    floor.material.color = Color::new(1., 0.9, 0.9);
    floor.material.specular = 0.;

    let mut left_wall = Sphere::new(2);
    left_wall.set_transform(
        translation(0., 0., 5.)*rotation_y(-PI/4.0)
        *rotation_x(PI/2.0)*scaling(10., 0.01, 10.)
    );
    left_wall.material = floor.material;

    let mut right_wall = Sphere::new(3);
    right_wall.set_transform(
        translation(0., 0., 5.)*rotation_y(PI/4.0)
        *rotation_x(PI/2.0)*scaling(10., 0.01, 10.)
    );
    right_wall.material = floor.material;

    let mut middle = Sphere::new(4);
    middle.set_transform(
        translation(-0.5, 1., 0.5)
    );
    middle.material.color = Color::new(0.1, 1., 0.5);
    middle.material.diffuse = 0.7;
    middle.material.specular = 0.3;

    let mut right = Sphere::new(5);
    right.set_transform(
        translation(1.5, 0.5, -0.5)*scaling(0.5, 0.5, 0.5)
    );
    right.material.color = Color::new(0.5, 1.0, 0.1);
    right.material.diffuse = 0.7;
    right.material.specular = 0.3;

    let mut left = Sphere::new(6);
    left.set_transform(
        translation(-1.5, 0.33, -0.75)*scaling(0.33, 0.33, 0.33)
    );
    left.material.diffuse = 0.7;
    left.material.specular = 0.3;

    let world = World{
        objects: vec![floor, left_wall, right_wall, middle, right, left],
        lights: vec![PointLight::new(Point::new(-10., 10., -10.), Color::new(1., 1., 1.))],
    };

    let mut camera = Camera::new(1000, 500, PI/3.0);
    camera.transform = view_transform(
        Point::new(0., 1.5, -5.), 
        Point::new(0., 1., 0.),
        Vector::new(0., 1., 0.)
    );

    let canvas = camera.render(&world);

    fs::write("sphere_scene_shadowed.ppm", canvas.to_ppm())?;
    Ok(())
}
