use linear_regression::{ok_or_exit, read_csv, LinearRegresser, write_csv, get_first_arg_or};

fn main() {
    let file = ok_or_exit!(get_first_arg_or("missing input file"));
    let mut lr = LinearRegresser::default();
    lr.data = ok_or_exit!(read_csv(file)).into();
    let (i, dev, pred) = lr.train();
    ok_or_exit!(write_csv("trained_data.csv", &pred));
    println!("predicted formula: `{} + {} * km` after {i} iterations with a deviation of {dev}", pred.theta_0, pred.theta_1);
}