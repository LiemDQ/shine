use std::cmp::PartialEq;
use std::ops::{Add, Sub, Mul};
use crate::utils::float_eq;

const MAX_PPM_LINE_WIDTH : usize = 70;

#[derive(Debug, Clone)]
pub struct Color {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
}

impl Color {
    pub const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { red: r, green: g, blue: b }
    }

    pub fn clamp(&mut self, min_val: f64, max_val: f64) -> &mut Self {
        self.red = self.red.clamp(min_val, max_val);
        self.green = self.green.clamp(min_val, max_val);
        self.blue = self.blue.clamp(min_val, max_val);
        self
    }

    pub fn round(&mut self) -> &mut Self {
        self.red = self.red.round();
        self.green = self.green.round();
        self.blue = self.blue.round();
        self
    }
    
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        float_eq(self.red, other.red) &&
        float_eq(self.green, other.green) &&
        float_eq(self.blue, other.blue)
    }
}

impl Add for Color {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            red: self.red + rhs.red,
            green: self.green + rhs.green,
            blue: self.blue + rhs.blue,
        }
    }
}

impl Sub for Color {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            red: self.red - rhs.red,
            green: self.green - rhs.green,
            blue: self.blue - rhs.blue,
        }
    }
}

impl Mul<f64> for Color {
    type Output = Color;
    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output {
            red: self.red * rhs,
            green: self.green * rhs,
            blue: self.blue * rhs,
        }
    }
}

impl Mul<Color> for f64 {
    type Output = Color;
    fn mul(self, rhs: Color) -> Self::Output {
        Color {
            red: rhs.red * self,
            green: rhs.green * self,
            blue: rhs.blue * self,
        }
    }
}

impl Mul for Color {
    type Output = Color;
    /// Hadamard product (element-wise multiplication)
    fn mul(self, rhs: Self) -> Self::Output {
        Self::Output {
            red: self.red * rhs.red,
            green: self.green * rhs.green,
            blue: self.blue * rhs.blue,
        }
    }
}

pub struct Canvas {
    w: usize,
    h: usize,
    screen: Vec<Vec<Pixel>>,
    max_pixel_value: f64,
    
}

impl Canvas {
    pub fn new( width: usize, height: usize) -> Self {
        let mut row = Vec::new();
        row.reserve(width);
        //initialize with white pixels
        row.resize(width, Pixel::new(0., 0., 0.));
        
        let mut screen = Vec::new();
        screen.reserve(height);
        screen.resize(height,row);

        Self { h: height, w: width, screen: screen, max_pixel_value: 255.0 }
    }

    pub fn height(&self) -> usize {
        self.h
    }

    pub fn width(&self) ->  usize {
        self.w
    }

    pub fn write_pixel(&mut self, x: usize, y: usize, c: Color) {
        self.screen[y][x] = c;
    }

    pub fn pixel_at(&self, x: usize, y: usize) -> &Pixel {
        &self.screen[y][x]
    }

    pub fn pixel_at_mut(&mut self, x: usize, y: usize) -> &mut Pixel {
        &mut self.screen[y][x]
    }

    pub fn to_ppm(&self) -> String {
        let mut header = format!("P3\n{} {}\n{}\n", self.w, self.h, self.max_pixel_value);
        for row in &self.screen {
            let mut line = String::new();
            line.reserve(MAX_PPM_LINE_WIDTH);
            let mut first = true;
            let mut sub = 0;
            for pixel in row {
                let px = self.scale(&pixel);
                //this is ugly but as there are only 3 colors it is ok
                if !first {
                    if line.len() - sub + 4 > MAX_PPM_LINE_WIDTH {
                        line.push('\n');
                        sub += line.len() - sub;
                    } else {
                        line.push(' ');
                    }
                }
                line.push_str(&format!("{}", px.red));

                if line.len() - sub + 4 > MAX_PPM_LINE_WIDTH {
                    line.push('\n');
                    sub += line.len() - sub;
                } else {
                    line.push(' ');
                }
                line.push_str(&format!("{}", px.green));

                if line.len() - sub + 4 > MAX_PPM_LINE_WIDTH {
                    line.push('\n');
                    sub += line.len() - sub;
                } else {
                    line.push(' ');
                }
                line.push_str(&format!("{}", px.blue));

                first = false;
            }
            header.push_str(&line.to_string());
            header.push('\n'); //last character must be a newline for compatibility with certain programs e.g. ImageMagick
        }
        header
    }

    pub fn fill(&mut self, pixel: Pixel) -> &mut Self {
        for row in &mut self.screen {
            row.fill(pixel.clone());
        }
        self
    }

    fn scale(&self,  pixel: &Pixel) -> Pixel {
        let mut result = pixel.clone();
        result = result * self.max_pixel_value;
        result.clamp(0.0, self.max_pixel_value).round();
        result
    }

}

/// Type alias for now, could be changed to something else later.
pub type Pixel = Color;

#[test]
fn create_color(){
    let color = Color { red: -0.5, green: 0.4, blue: 1.7};
    assert_eq!(color, Color::new(-0.5, 0.4, 1.7));
}

#[test]
fn add_colors() {
    let c1 = Color::new(0.9, 0.6, 0.75);
    let c2 = Color::new(0.7, 0.1, 0.25);
    assert_eq!(c1 + c2, Color::new(1.6, 0.7, 1.0));
}

#[test]
fn sub_colors(){
    let c1 = Color::new(0.9, 0.6, 0.75);
    let c2 = Color::new(0.7, 0.1, 0.25);
    assert_eq!(c1 - c2, Color::new(0.2, 0.5, 0.5));
}

#[test]
fn mult_color_by_scalar(){
    let c = Color::new(0.2, 0.3, 0.4);
    assert_eq!(c * 2.0, Color::new(0.4, 0.6, 0.8));
}

#[test]
fn mult_color_by_color(){
    let c1 = Color::new(1.0, 0.2, 0.4);
    let c2 = Color::new(0.9, 1.0, 0.1);
    assert_eq!(c1*c2, Color::new(0.9, 0.2, 0.04));
}

#[test]
fn initialize_canvas(){
    let canvas = Canvas::new(10, 20);
    assert_eq!(canvas.w, 10);
    assert_eq!(canvas.h, 20);
    for row in canvas.screen {
        for p in row {
            assert_eq!(p, Color::new(0.,0.,0.));
        }
    }
}

#[test]
fn access_pixels() {
    let red = Color::new(1., 0., 0.);
    let mut c = Canvas::new(10, 20);
    c.write_pixel(2, 3, red.clone());
    
    assert_eq!(*c.pixel_at(2, 3), red);
}

#[test]
fn create_ppm_from_canvas(){
    let mut c = Canvas::new(5, 3);
    let c1 = Color::new(1.5, 0., 0.);
    let c2 = Color::new(0., 0.5, 0.);
    let c3 = Color::new(-0.5, 0., 1.);
    c.write_pixel(0, 0, c1);
    c.write_pixel(2,1, c2);
    c.write_pixel(4,2, c3);
    let ppm = c.to_ppm();
    assert_eq!(ppm, 
"P3
5 3
255
255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 128 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 255
"
    )
}

#[test]
fn create_ppm_with_line_overwrap(){
    let mut c = Canvas::new(10, 2);
    c.fill(Pixel::new(1.0, 0.8, 0.6));
    let ppm = c.to_ppm();
    assert_eq!(ppm,
"P3
10 2
255
255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153
255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153
"
    );
}