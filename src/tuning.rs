use std::io;
use std::io::Write;
use serde::{Serialize, Deserialize};

use chatclm::backend::dataset::Dataset;
use chatclm::backend::trainer::train_model;
use chatclm::backend::training_options::TrainingOptions;


#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TuningParameters {
    d: u32,
    f: u32,
    k: f64,
    compression_level: u32,
    dataset_size: u32,
    dictionary_size_percentage: f64
}

#[derive(Serialize, Deserialize)]
struct TuningMetrics {
    val_bpt: f64,
    val_bpt_stderr: f64,
    train_bpt: f64,
    train_pbt_stderr: f64,
    val_inf_gain: f64,
    val_inf_gain_stderr: f64,
    train_inf_gain: f64,
    train_inf_gain_stderr: f64,
    training_time: f64,
    dictionary_size: usize
}

fn main() {

    // read the dataset
    let dataset = Dataset::load_or_compute("dataset.checkpoint");
    
    // read the parameters from stdin
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let params: TuningParameters = serde_json::from_str(&input).unwrap();


    let (train, test) = dataset.split_train_test(0.9);
    
    let shrunk_train = dataset.shrink_to_size(params.dataset_size as usize);
    
    
    // train the model
    let start_time = std::time::Instant::now();
    let default_params = TrainingOptions::default();
    let training_paramters = TrainingOptions {
        accel: default_params.accel,
        d: params.d,
        f: params.f,
        k: params.k as u32,
        shrink_dict: default_params.shrink_dict,
        shrink_dict_max_regression: default_params.shrink_dict_max_regression,
        split_point: default_params.split_point,
        steps: default_params.steps,
        nb_threads: 8,
        compression_level: params.compression_level,
        dictionary_size_percentage: params.dictionary_size_percentage 
    };

    let model = train_model(train.get_data(), training_paramters);

    let time = start_time.elapsed().as_secs_f64();

    // evaluate the model
    let val_accuracy = model.average_bytes_per_token(&test);
    let train_accuracy = model.average_bytes_per_token(&shrunk_train);

    let val_inf_gain = model.average_information_gain(&test);
    let train_inf_gain = model.average_information_gain(&shrunk_train);

    // write the metrics to stdout
    let metrics = TuningMetrics {
        val_bpt: val_accuracy.0,
        val_bpt_stderr: val_accuracy.1,
        train_bpt: train_accuracy.0,
        train_pbt_stderr: train_accuracy.1,
        val_inf_gain: val_inf_gain.0,
        val_inf_gain_stderr: val_inf_gain.1,
        train_inf_gain: train_inf_gain.0,
        train_inf_gain_stderr: train_inf_gain.1,
        dictionary_size: model.get_dictionary_size(),
        training_time: time
    };
    
    // flush stdout
    io::stdout().flush().unwrap();
    let output = serde_json::to_string(&metrics).unwrap();
    println!("{}", output);
    io::stdout().flush().unwrap();
}