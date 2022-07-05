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

use crate::{ray::*, canvas::Color};

fn main() -> io::Result<()> {
    use std::f64::consts::PI;
    let mut floor = Box::new(Plane::new());
    floor.mut_material().pattern = CheckerPattern::new(Color::black(), Color::white());
    floor.mut_material().reflective = 0.3;

    let mut right_wall = Box::new(Plane::new());
    right_wall.set_transform(translation(0., 0.0, 10.0)*rotation_y(-PI/4.0)*rotation_x(PI/2.0));
    right_wall.mut_material().pattern = CheckerPattern::new(Color::black(), Color::white());

    let mut left_wall = Box::new(Plane::new());
    left_wall.set_transform(translation(0., 0., 10.0)*rotation_y(PI/4.0)*rotation_x(PI/2.0));
    left_wall.mut_material().pattern = CheckerPattern::new(Color::black(), Color::white());

    let mut right_front_wall = Box::new(Plane::new());
    right_front_wall.set_transform(translation(0., 0.0, -25.0)*rotation_y(-PI/4.0)*rotation_x(PI/2.0));
    right_front_wall.mut_material().pattern = CheckerPattern::new(Color::black(), Color::white());

    let mut left_front_wall = Box::new(Plane::new());
    left_front_wall.set_transform(translation(0., 0., -25.0)*rotation_y(PI/4.0)*rotation_x(PI/2.0));
    left_front_wall.mut_material().pattern = CheckerPattern::new(Color::black(), Color::white());
    
    let mut middle = Box::new(Sphere::new());
    middle.set_transform(
        translation(-0.5, 1., 0.5)
    );
    middle.mut_material().pattern = SolidPattern::new(Color::red());
    middle.mut_material().diffuse = 0.7;
    middle.mut_material().specular = 0.3;
    middle.mut_material().transparency = 0.7;
    middle.mut_material().refractive_index = 1.1;
    middle.mut_material().reflective = 0.4;
    
    // middle.mut_material().pattern = CheckerPattern::new(Color::red(), Color::white());
    // middle.mut_material().pattern.set_transform(scaling(0.25, 0.25, 0.25));

    let mut right = Box::new(Sphere::new());
    right.set_transform(
        translation(1.5, 0.5, -0.5)*scaling(0.5, 0.5, 0.5)
    );
    right.mut_material().pattern = SolidPattern::new(Color::new(0.5, 1.0, 0.1));
    right.mut_material().diffuse = 0.7;
    right.mut_material().specular = 0.3;

    let mut left = Box::new(Sphere::new());
    left.set_transform(
        translation(-1.5, 0.33, -0.75)*scaling(0.33, 0.33, 0.33)
    );
    left.mut_material().diffuse = 0.7;
    left.mut_material().specular = 0.3;

    let mut behind = Box::new(Sphere::new());
    behind.set_transform(
        translation(1.5, 1.5, 3.0)*scaling(1.5, 1.5, 1.5)
    );
    behind.mut_material().pattern = SolidPattern::new(Color::blue());
    behind.mut_material().transparency = 0.4;
    behind.mut_material().refractive_index = 1.2;


    let world = World{
        objects: vec![floor, middle, right, left, right_wall, left_wall, right_front_wall, left_front_wall, behind],
        lights: vec![PointLight::new(Point::new(-10., 10., -5.), Color::new(1., 1., 1.))],
    };

    let mut camera = Camera::new(1000, 500, PI/3.0);
    camera.transform = view_transform(
        Point::new(0., 1.5, -5.), 
        Point::new(0., 1., 0.),
        Vector::new(0., 1., 0.)
    );

    let canvas = camera.render(&world);

    fs::write("sphere_checkered_floor.ppm", canvas.to_ppm())?;
    Ok(())
}
