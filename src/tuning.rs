use std::io;
use std::io::Write;
use serde::{Serialize, Deserialize};

use chatclm::backend::dataset::Dataset;
use chatclm::backend::trainer::train_model;
use chatclm::backend::training_options::TrainingOptions;


#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TuningParameters {
    accel: u32,
    d: u32,
    f: u32,
    k: f64,
    shrink_dict: u32,
    shrink_dict_max_regression: f64,
    split_point: f64,
    steps: f64,
    compression_level: u32
}
//{"accel": 8, "d": 6, "f": 6, "k": 100, "shrinkDict": 1, "shrinkDictMaxRegression": 10782724, "splitPoint": 0.8800316450822921, "steps": 5, "compressionLevel": 12}

#[derive(Serialize, Deserialize)]
struct TuningMetrics {
    val_bpt: f64,
    val_bpt_stderr: f64,
    train_bpt: f64,
    train_pbt_stderr: f64,
    training_time: f64
}

fn main() {
    // read the parameters from stdin
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let params: TuningParameters = serde_json::from_str(&input).unwrap();

    // read the dataset
    let dataset = Dataset::load_or_compute("dataset.checkpoint");

    let (train, test) = dataset.split_train_test(0.8);

    // train the model
    let start_time = std::time::Instant::now();

    let training_paramters = TrainingOptions {
        accel: params.accel,
        d: params.d,
        f: params.f,
        k: params.k as u32,
        shrink_dict: params.shrink_dict,
        shrink_dict_max_regression: params.shrink_dict_max_regression as u32,
        split_point: params.split_point,
        steps: params.steps as u32,
        compression_level: params.compression_level,
        nb_threads: 1
    };

    let model = train_model(train.get_data(), training_paramters);

    let time = start_time.elapsed().as_secs_f64();

    // evaluate the model
    let val_accuracy = model.average_bytes_per_token(&test);
    let train_accuracy = model.average_bytes_per_token(&train);

    // write the metrics to stdout
    let metrics = TuningMetrics {
        val_bpt: val_accuracy.0,
        val_bpt_stderr: val_accuracy.1,
        train_bpt: train_accuracy.0,
        train_pbt_stderr: train_accuracy.1,
        training_time: time
    };
    
    // flush stdout
    io::stdout().flush().unwrap();
    let output = serde_json::to_string(&metrics).unwrap();
    println!("{}", output);
    io::stdout().flush().unwrap();
}