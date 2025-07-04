trainer:
	cargo run --package linear_regression --release --bin trainer -- data.csv

predictor:
	cargo run --package linear_regression --release --bin predictor -- 176000

bonus:
	cargo run --package linear_regression --release --bin linear_regression

clean:
	rm -rf target

tmp_rust:
	#binaries:  $CARGO_HOME/bin
	#lib:       $RUSTUP_HOME/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust
	RUSTUP_HOME=/tmp/rustup CARGO_HOME=/tmp/cargo bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"