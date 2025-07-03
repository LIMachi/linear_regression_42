use eframe::egui;
use egui::{InputState, PointerButton, Pos2, Ui};
use egui_plot::{Plot, PlotPoints, Points, Line, PlotPoint, PlotUi, PlotResponse};
use linear_regression::{get_first_arg_or, read_csv, Entry, LinearRegresser, TrainedPredictor};

fn main() {
    let mut linear_regresser = LinearRegresser::default();
    if let Ok(file) = get_first_arg_or(()) {
        if let Ok(csv) = read_csv(file) {
            linear_regresser.data = csv.into();
        }
    };
    let mut app = App {
        linear_regresser,
        predictor: TrainedPredictor::default(),
        steps: 0,
        deviation: 0.,

        lr_input: String::new(),
        lt_input: String::new(),
        it_input: String::new(),
    };
    app.lr_input = format!("{:.9}", app.linear_regresser.learning_rate).trim_end_matches(&['0', ' ']).to_string();
    app.lt_input = format!("{:.9}", app.linear_regresser.delta_threshold).trim_end_matches(&['0', ' ']).to_string();
    app.it_input = app.linear_regresser.iterations.to_string();
    app.compute();
    eframe::run_native(
        "Linear Regression",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(app))),
    ).unwrap();
}


struct App {
    linear_regresser: LinearRegresser,
    predictor: TrainedPredictor,
    steps: usize,
    deviation: f64,

    lr_input: String,
    lt_input: String,
    it_input: String,
}

impl App {
    fn compute(&mut self) {
        (self.steps, self.deviation, self.predictor) = self.linear_regresser.train();
    }

    fn controls(&mut self, ui: &mut Ui) {
        ui.heading("Settings");

        let mut recompute = false;

        ui.label("Learning Rate:");
        if ui.text_edit_singleline(&mut self.lr_input).lost_focus() {
            if let Ok(val) = self.lr_input.parse::<f64>() {
                self.linear_regresser.learning_rate = val;
                self.lr_input = format!("{:.9}", self.linear_regresser.learning_rate).trim_end_matches(&['0', ' ']).to_string();
                recompute = true;
            }
        }

        ui.label("Learning Threshold:");
        if ui.text_edit_singleline(&mut self.lt_input).lost_focus() {
            if let Ok(val) = self.lt_input.parse::<f64>() {
                self.linear_regresser.delta_threshold = val;
                self.lt_input = format!("{:.9}", self.linear_regresser.delta_threshold).trim_end_matches(&['0', ' ']).to_string();
                recompute = true;
            }
        }

        ui.label("Iterations:");
        if ui.text_edit_singleline(&mut self.it_input).lost_focus() {
            if let Ok(val) = self.it_input.parse::<usize>() {
                self.linear_regresser.iterations = val;
                self.it_input = self.linear_regresser.iterations.to_string();
                recompute = true;
            }
        }

        ui.separator();

        ui.heading("Results:");

        ui.label(format!("iterations: {:.13}", self.steps));
        ui.label(format!("Theta 0:    {:.10}", self.predictor.theta_0));
        ui.label(format!("Theta 1:    {:.10}", self.predictor.theta_1));
        ui.label(format!("Deviation:  {:.10}", self.deviation));

        if recompute {
            self.compute();
        }
    }
    
    fn plot(&mut self, ui: &mut PlotUi) {
        let points = PlotPoints::Owned(self.linear_regresser.data
            .iter()
            .map(|e| PlotPoint::from([e.km, e.price]))
            .collect::<Vec<PlotPoint>>());

        ui.points(Points::new("Data", points).radius(2.));

        let min_km = self.linear_regresser.data.iter().map(|e| e.km).fold(f64::INFINITY, f64::min);
        let max_km = self.linear_regresser.data.iter().map(|e| e.km).fold(f64::NEG_INFINITY, f64::max);

        let step = (max_km - min_km) / 100.0;
        let line_points = PlotPoints::Owned((0..=100)
            .map(|i| {
                let x = min_km + i as f64 * step;
                let y = self.predictor.predict(x);
                [x, y]
            })
            .map(PlotPoint::from)
            .collect::<Vec<PlotPoint>>());

        ui.line(Line::new("Prediction", line_points));
    }
    
    fn mouse(&mut self, ui: &mut Ui, plot_view: &PlotResponse<()>, pos: Pos2) {
        if plot_view.response.ctx.input(|i| i.pointer.button_clicked(PointerButton::Primary)) {
            let pos = &plot_view.transform.value_from_position(pos);
            self.linear_regresser.data.push(Entry {
                km: pos.x,
                price: pos.y,
            });
            self.compute();
        }
        if plot_view.response.ctx.input(|i| i.pointer.button_clicked(PointerButton::Middle)) {
            let mut best = None;
            for (i, e) in self.linear_regresser.data.iter().enumerate() {
                let e = plot_view.transform.position_from_point(&PlotPoint::from([e.km, e.price]));
                let dx = e.x - pos.x;
                let dy = e.y - pos.y;
                let d = dx * dx + dy * dy;
                if d <= 25. { //we are within 5 pixels of this point
                    if let Some((b, bd)) = best {
                        if d < bd {
                            best = Some((i, d));
                        }
                    } else {
                        best = Some((i, d));
                    }
                }
            }
            if let Some((i, _)) = best {
                self.linear_regresser.data.remove(i);
                self.compute();
            }
        }
    }
    
    fn input(&mut self, input: &InputState) {
            for file in &input.raw.dropped_files {
                if let Some(path) = &file.path {
                    if let Ok(csv) = read_csv(path) {
                        self.linear_regresser.data = csv.into();
                        self.predictor = TrainedPredictor::default();
                        self.compute();
                    }
                }
            }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(250.)
            .show(ctx, |ui| self.controls(ui));

        egui::CentralPanel::default().show(ctx, |ui| {
            let plot_view = Plot::new("Regression")
                .x_axis_label("km")
                .y_axis_label("price")
                .label_formatter(|name, value| format!("km: {:.3}\nprice: {:.3}", value.x, value.y))
                .show(ui, |ui| self.plot(ui));
            if let Some(pointer_pos) = ui.ctx().pointer_interact_pos() {
                if plot_view.response.rect.contains(pointer_pos) {
                    self.mouse(ui, &plot_view, pointer_pos);
                }
            }
        });
        
        ctx.input(|i| self.input(i));
    }
}