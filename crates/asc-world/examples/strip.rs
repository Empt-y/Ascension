//! Prints ground height minus water level over a small area (debugging rivers):
//! `cargo run --release -p asc-world --example strip -- x0 z0 x1 z1 step`.
fn main() {
    let a: Vec<f32> = std::env::args().skip(1).map(|v| v.parse().unwrap()).collect();
    let wg = asc_world::WorldGen::new(1337);
    let mut z = a[1];
    while z <= a[3] {
        let mut line = format!("z{z:5.0} ");
        let mut x = a[0];
        while x <= a[2] {
            let (h, w) = wg.height_and_water(x, z);
            line += &match w {
                Some(l) => if std::env::var("ABS").is_ok() { format!("{:5.1}", l) } else { format!("{:+4.1}", h - l) },
                None => "  . ".to_string(),
            };
            x += a[4];
        }
        println!("{line}");
        z += a[4];
    }
}
