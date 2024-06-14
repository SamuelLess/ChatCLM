use rand::Rng;
use rand::seq::SliceRandom;

use crate::backend::clm_model::ClmModel;
use crate::backend::dataset::Dataset;

const SAMPLES: usize = 10000;

impl<'a> ClmModel<'a> {
    pub fn average_bytes_per_token(&self, test_data: &Dataset) -> (f64, f64) {
        println!("Test data size: {}", test_data.get_data().len());
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
            values.push(compressed as i32 - compressed_prompt.len() as i32)
        }

        let average = values.iter().sum::<i32>() as f64 / values.len() as f64;
        let standard_deviation = values.iter().map(|x| (x - values.iter().sum::<i32>() / values.len() as i32).pow(2)).sum::<i32>() as f64 / values.len() as f64;
        let standard_error = standard_deviation / (values.len() as f64).sqrt();
        let confidence_interval = 1.96 * standard_error;
        println!("Confidence interval: [{} – {}]", average - confidence_interval, average + confidence_interval);

        (average, standard_error)
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

        let model = train_model(&Vec::new(), TrainingOptions::new());
        let (initial_avg,_) = model.average_bytes_per_token(&training_data);

        let trained_model = train_model(&training_data.get_data(), TrainingOptions::new());
        let (trained_avg, _) = trained_model.average_bytes_per_token(&training_data);

        assert!(trained_avg < initial_avg);

        println!("Initial avg: {}", initial_avg);
        println!("Trained avg: {}", trained_avg);
    }
}