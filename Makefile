trainer:
	cargo run --package linear_regression --release --bin trainer -- data.csv

predictor:
	cargo run --package linear_regression --release --bin predictor -- 176000

bonus:
	cargo run --package linear_regression --release --bin linear_regression