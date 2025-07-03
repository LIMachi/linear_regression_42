use linear_regression::{ok_or_exit, get_first_arg_or, read_csv, TrainedPredictor};

fn main() {
    let predictor: TrainedPredictor = ok_or_exit!(read_csv("trained_data.csv").and_then(|v| v.get(0).ok_or("missing training data in file".to_string()).copied()));
    let kms = ok_or_exit!(get_first_arg_or("missing mileage").and_then(|v| v.parse::<f64>().map_err(|_| "invalid value format")));
    println!("predicted price: {}", predictor.predict(kms));
}