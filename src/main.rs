#[macro_use]
mod utils;
mod coords;
mod canvas;
mod matrix;
mod transforms;

use coords::{Point, Vector, Coord};
use canvas::{Canvas, Pixel};
use std::{fs, io};
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

fn main() -> io::Result<()> {
    let start = Point::new(0., 1., 0.);
    let velocity = Vector::new(1., 1.8, 0.).normalize() * 3.25;
    let mut proj = Projectile {position: start, velocity: velocity};

    let envir = Environment { gravity: Vector::new(0., -0.1, 0.), wind: Vector::new(-0.01, 0., 0.) };
    let mut count = 0;

    let mut canvas = Canvas::new(900, 550);
    
    while proj.position.y > 0.0 {
        //println!("{:?}", &proj.position);
        tick(& envir, &mut proj);
        imprint_proj_on_canvas(&mut canvas, &proj);
        count += 1;
    }
    
    fs::write("trajectory.ppm", canvas.to_ppm())?;
    println!("Took {} ticks to hit the ground.", count);
    Ok(())
}
