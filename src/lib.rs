use std::env;
use std::path::Path;
use csv::{ReaderBuilder, WriterBuilder};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

#[derive(Serialize, Deserialize, Debug, Default, Copy, Clone)]
pub struct TrainedPredictor {
    pub theta_0: f64,
    pub theta_1: f64,
}

#[derive(Deserialize, Debug)]
pub struct Entry {
    pub km: f64,
    pub price: f64
}

pub fn mean_std<I: Iterator<Item=f64> + Clone>(iter: I) -> (f64, f64) {
    let len = iter.clone().count() as f64;
    let mean = iter.clone().sum::<f64>() / len;
    let squared_deviation = iter.map(|v| (v - mean) * (v - mean)).sum::<f64>() / len;
    (mean, squared_deviation.sqrt())
}

#[derive(Debug)]
pub struct LinearRegresser {
    pub data: Vec<Entry>,
    pub learning_rate: f64,
    pub iterations: usize,
    pub delta_threshold: f64,
}

impl Default for LinearRegresser {
    fn default() -> Self {
        Self {
            data: Default::default(),
            learning_rate: 0.1,
            iterations: 1000,
            delta_threshold: 0.0001,
        }
    }
}

impl TrainedPredictor {
    pub fn predict(&self, millage: f64) -> f64 {
        self.theta_0 + self.theta_1 * millage
    }

    pub fn delta(&self, other: &Self) -> f64 {
        let d1 = self.theta_1 - other.theta_1;
        (self.theta_0 - other.theta_0).abs() + d1 * d1
    }
}

impl LinearRegresser {
    pub fn train(&mut self) -> (usize, f64, TrainedPredictor) {
        if self.data.len() == 0 || self.learning_rate <= 0.0 || self.iterations <= 0 {
            return (0, 0., TrainedPredictor::default());
        }
        let (km_mean, km_std) = mean_std(self.data.iter().map(|e| e.km));
        let (price_mean, price_std) = mean_std(self.data.iter().map(|e| e.price));
        let mut prediction = TrainedPredictor {
            theta_0: price_mean,
            theta_1: price_std / km_std,
        };
        let data: Vec<Entry> = self.data.iter().map(|e| Entry {
            km: (e.km - km_mean) / km_std,
            price: (e.price - price_mean) / price_std,
        }).collect();
        let weight = 1f64 / data.len() as f64; // 1 / m
        let mut early_return = None;
        for i in 0..self.iterations {
            let mut sum_errors_price = 0.;
            let mut sum_errors_mileage = 0.;
            for entry in &data { //Epsilon { i = 0, m - 1 }
                let predicted = prediction.predict(entry.km); //estimatePrice(mileage[i])
                let error_price = predicted - entry.price; //(estimatePrice(mileage[i]) − price[i])
                sum_errors_price += error_price;
                sum_errors_mileage += error_price * entry.km; //((estimatePrice(mileage[i]) − price[i]) * mileage[i])
            }
            let t = TrainedPredictor {
                theta_0: prediction.theta_0 - self.learning_rate * weight * sum_errors_price, //learningRate * (1 / m) * Epsilon { i = 0, m - 1 } (estimatePrice(mileage[i]) − price[i])
                theta_1: prediction.theta_1 - self.learning_rate * weight * sum_errors_mileage, //learningRate * (1 / m) * Epsilon { i = 0, m - 1 } ((estimatePrice(mileage[i]) − price[i]) * mileage[i])
            };
            let delta = t.delta(&prediction);
            prediction = t;
            if self.delta_threshold > 0.0 && delta < self.delta_threshold {
                early_return = Some(i);
                break;
            }
        }
        let theta_1 = (price_std / km_std) * prediction.theta_1;
        let final_predictor = TrainedPredictor {
            theta_1,
            theta_0: price_mean - theta_1 * km_mean + price_std * prediction.theta_0,
        };
        (
            early_return.unwrap_or(self.iterations),
            {
                let mut acc = 0.0;
                for entry in &self.data {
                    let d = final_predictor.predict(entry.km) - entry.price;
                    acc += d * d;
                }
                (acc / self.data.len() as f64).sqrt()
            },
            final_predictor
        )
    }
}

pub fn get_first_arg_or<T>(err: T) -> Result<String, T> {
    env::args().skip(1).next().ok_or(err)
}

pub fn read_csv<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<Vec<T>, String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| e.to_string())?;
    Ok(reader.deserialize().filter_map(Result::ok).collect::<Vec<T>>())
}

pub fn write_csv<T: Serialize, P: AsRef<Path>>(path: P, data: &T) -> Result<(), String> {
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| e.to_string())?;
    writer.serialize(data).map_err(|e| e.to_string())?;
    Ok(())
}

#[macro_export]
macro_rules! ok_or_exit {
    ($result:expr) => {
        match $result {
            Ok(val) => val,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(0);
            }
        }
    };
}