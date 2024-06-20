use num::Num;
use num::pow::Pow;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::backend::clm_model::ClmModel;
use crate::backend::dataset::Dataset;
use crate::backend::{MAX_TOKEN, Token};

const SAMPLES: usize = 20000;


fn average_with_error(values: Vec<f64>) -> (f64, f64){
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let standard_deviation = values.iter().map(|x| (x - values.iter().sum::<f64>() / values.len() as f64).pow(2)).sum::<f64>() / values.len() as f64;
    let standard_error = standard_deviation / (values.len() as f64).sqrt();

    (average, standard_error)
}

fn fmax<T: Num + PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

impl<'a> ClmModel<'a> {
    pub fn average_bytes_per_token(&self, test_data: &Dataset) -> (f64, f64) {
        let mut values = Vec::new();

        let mut rng = rand::thread_rng();
        for _ in 0..SAMPLES {
            let sentence_index = rng.gen_range(0..test_data.get_data().len());
            let sentence = &test_data.get_data()[sentence_index];
            if sentence.len() < 7 {
                continue;
            }

            let pos = rng.gen_range(5..sentence.len());

            let prompt = sentence[..pos].to_vec();
            let next_token = sentence[pos];
            let compressed_prompt = self.compress(&prompt);
            let compressed = self.compress_together(&prompt, &vec![next_token]);
            values.push(compressed as f64 - compressed_prompt.len() as f64)
        }

        average_with_error(values)
    }

    pub fn average_information_gain(&self, test_data: &Dataset) -> (f64, f64){
        let mut values = Vec::new();

        let mut rng = rand::thread_rng();
        for _ in 0..SAMPLES {
            let sentence_index = rng.gen_range(0..test_data.get_data().len());
            let sentence = &test_data.get_data()[sentence_index];
            if sentence.len() < 7 {
                continue;
            }

            let pos = rng.gen_range(5..sentence.len());

            let prompt = sentence[..pos].to_vec();
            let next_token = sentence[pos];
            let random_token = rng.gen_range(0..MAX_TOKEN) as Token;

            let compressed_prompt = self.compress(&prompt).len() as f64;
            let compressed_truth = self.compress_together(&prompt, &vec![next_token]) as f64;
            let compressed_random = self.compress_together(&prompt, &vec![random_token]) as f64;

            let truth_bytes_added = compressed_truth - compressed_prompt;
            let random_bytes_added = compressed_random - compressed_prompt;

            values.push(random_bytes_added/fmax(0.1, truth_bytes_added));
        }

        average_with_error(values)
    }
}


mod tests {
    use crate::backend::dataset::Dataset;
    use crate::backend::tests::random_tokens;
    use crate::backend::trainer::train_model;
    use crate::backend::training_options::TrainingOptions;

    #[test]
    fn average_bytes_per_token_improves() {
        let random_data = random_tokens(300);
        let training_data = Dataset::from_data((0..10).map(|_| random_data.clone()).collect());

        let model = train_model(&Vec::new(), &TrainingOptions::new());
        let (initial_avg,_) = model.average_bytes_per_token(&training_data);

        let trained_model = train_model(&training_data.get_data(), &TrainingOptions::new());
        let (trained_avg, _) = trained_model.average_bytes_per_token(&training_data);

        assert!(trained_avg < initial_avg);

        println!("Initial avg: {}", initial_avg);
        println!("Trained avg: {}", trained_avg);
    }

    #[test]
    fn untrained_model_has_average_information_gain_of_one() {
        let random_data = random_tokens(300);
        let testing_data = Dataset::from_data((0..10).map(|_| random_data.clone()).collect());

        let model = train_model(&Vec::new(), &TrainingOptions::new());
        let (initial_avg, initial_stderr) = model.average_information_gain(&testing_data);

        // 99%  confidence interval
        let confidence_interval = 2.576 * initial_stderr;
        println!("Initial avg: {}", initial_avg);
        println!("Standard error: {}", initial_stderr);
        println!("Confidence interval: [{} – {}]", initial_avg - confidence_interval, initial_avg + confidence_interval);

        assert!(0.99f64 <= initial_avg + confidence_interval);
        assert!(1.01f64 >= initial_avg - confidence_interval);

    }
    #[test]
    fn average_information_gain_is_greater_than_one() {
        let random_data = random_tokens(100);
        let training_data = Dataset::from_data((0..8).map(|_| random_data.clone()).collect());

        let model = train_model(&Vec::new(), &TrainingOptions::new());
        let (initial_avg,_) = model.average_information_gain(&training_data);

        let trained_model = train_model(&training_data.get_data(), &TrainingOptions::new());
        let (trained_avg, trained_stderr) = trained_model.average_information_gain(&training_data);

        assert!(trained_avg > 1f64);

        println!("Initial avg: {}", initial_avg);
        println!("Trained avg: {}", trained_avg);
        println!("Trained stderr: {}", trained_stderr);
    }
}