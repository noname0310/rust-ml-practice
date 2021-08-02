mod functions;

use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::view::ContinuousView;
use plotlib::style::{LineStyle};
use bevy::prelude::*;

fn make_plot<T: Into<String>>(vec: &Vec<f64>, f: fn(v: &Vec<f64>) -> Vec<f64>, colour: T) -> Plot {
    Plot::new(
        vec.iter()
        .zip(f(&vec))
        .map(|(x, y)| (x.clone(), y.clone()))
        .collect()
    ).line_style(
    LineStyle::new()
        .colour(colour),
    )
}

struct Position { x: f32, y: f32 }

fn print_position_system(query: Query<&Transform>) {
    for transform in query.iter() {
        println!("position: {:?}", transform.translation);
    }
}

struct Entity(u64);

fn hello_world() {
    println!("hello world!");
}

struct Person;

struct Name(String);

fn add_people(mut commands: Commands) {
    commands.spawn().insert(Person).insert(Name("Elaina Proctor".to_string()));
    commands.spawn().insert(Person).insert(Name("Renzo Hume".to_string()));
    commands.spawn().insert(Person).insert(Name("Zayna Nieves".to_string()));
}

fn greet_people(query: Query<&Name, With<Person>>) {
    for name in query.iter() {
        println!("hello {}!", name.0);
    }
}
pub struct HelloPlugin;

impl Plugin for HelloPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_startup_system(add_people.system())
            .add_system(hello_world.system())
            .add_system(greet_people.system());
    }
}

fn main() {
    let range_start = -80;
    let range_end = 80;
    let vec: Vec<f64> = (range_start..range_end).into_iter().map(|x| x as f64 / 10.).collect();

    let v = ContinuousView::new()
        .add(make_plot(&vec, functions::sigmoid_vec, "#DD3355"))
        .add(make_plot(&vec, functions::softmax_vec, "#00AA00"))
        .x_range((range_start / 10) as f64, (range_end / 10) as f64)
        .y_range(0., 1.)
        .x_label("x")
        .y_label("y");

    Page::single(&v).save("scatter.svg").unwrap();
    App::build()
        .add_plugins(DefaultPlugins)
        .add_plugin(HelloPlugin)
        .run();
}
