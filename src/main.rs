mod coords;

use coords::{Point, Vector, Coord};


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

fn main() {
    let envir = Environment { gravity: Vector::new(0., -0.1, 0.), wind: Vector::new(-0.01, 0., 0.) };
    let mut proj = Projectile {position: Point::new(0., 1.0, 0.), velocity: 8.0*Vector::new(1., 1., 0.).normalize()};
    let mut count = 0;

    while proj.position.y > 0.0 {
        println!("{:?}", &proj.position);
        tick(& envir, &mut proj);
        count += 1;
    }

    println!("Took {} ticks to hit the ground.", count);
}
