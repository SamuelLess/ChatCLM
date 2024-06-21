use indicatif::ProgressIterator;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use chatclm::backend::dataset::Dataset;
use chatclm::backend::ensemble_model::EnsembleModel;
use chatclm::backend::MAX_TOKEN;
use chatclm::backend::Token;
use chatclm::backend::training_options::TrainingOptions;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TuningParameters {
    d: u32,
    f: u32,
    k: f64,
    compression_level: u32,
    dataset_size: u32,
    dictionary_size_percentage: f64,
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
    dictionary_size: usize,
}
/*
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
        dictionary_size_percentage: params.dictionary_size_percentage, 
        ensemble_size: 1
    };

    let model = train_model(train.get_data(), &training_paramters);

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
}*/

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();


    let retrain = false;
    let trained_model = if retrain {
        println!("Reading dataset");
        let dataset = Dataset::load_or_compute("dataset.checkpoint");
        println!("Training models");

        let mut options = TrainingOptions::default();
        options.ensemble_size = 20;
        let trained_model = EnsembleModel::train(dataset, &options);
        trained_model.save_checkpoint("ensemble.checkpoint");
        trained_model
    } else {
        EnsembleModel::from_checkpoint("ensemble.checkpoint")
    };


    // Tokenize "The quick brown fox jumps over the lazy dog"
    let prompt = "The quick brown fox jumps over the lazy";
    let mut prompt_tokens = Dataset::tokenize(prompt);
    let mut size_before = trained_model.compressed_size(&prompt_tokens);

    for _ in 0..50 {
        println!("Prompt: {} ", Dataset::detokenize(prompt_tokens.clone()).as_str());

        let mut sizes: Vec<(Token, f64)> = (0..(Dataset::get_tokenizer().get_max_token()+1)).progress().par_bridge().map(|next_token| {
            let mut tokens = prompt_tokens.clone();
            tokens.push(next_token as Token);
            let size = trained_model.compressed_size(&tokens);
            println!("Size: {}", size);
            (next_token as Token, 1.0 / (size - size_before))
        }).collect();

        //shuffle
        sizes.shuffle(&mut rand::thread_rng());

        // sort descending
        sizes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // take only the first 200 tokens
        sizes.truncate(10);

        // subtract the mimimum likelihood
        let min_size = sizes.iter().map(|(_token, size)| *size).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        sizes.iter_mut().for_each(|(_token, size)| *size -= min_size);

        // normalize the distribution
        let sum: f64 = sizes.iter().map(|(_token, size)| *size).sum();
        sizes.iter_mut().for_each(|(_token, size)| *size /= sum);

        // print the sum of all sizes
        let sum: f64 = sizes.iter().map(|(_token, size)| *size).sum();
        println!("Sum: {}", sum);

        // print the ten smallest sizes
        for (token, size) in sizes.iter().take(10) {
            println!("Token: '{}' ({}), Size: {}", Dataset::detokenize(vec![*token]), *token as u64,  size);
        }


        // choose a token with probability proportional to the size
        let mut rng = rand::thread_rng();
        let (next_token, next_size) = sizes.choose_weighted(&mut rng, |(_token, size)| *size).unwrap();
        prompt_tokens.push(*next_token);
        size_before = *next_size;
    }
}