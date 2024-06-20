
use rayon::prelude::*;
use rusqlite::Connection;

use crate::backend::clm_model::ClmModel;
use crate::backend::dataset::Dataset;
use crate::backend::Token;
use crate::backend::trainer::train_model;
use crate::backend::training_options::TrainingOptions;

pub struct EnsembleModel<'a> {
    models: Vec<ClmModel<'a>>,
}

impl EnsembleModel<'_> {
    pub fn train(data: Dataset, options: &TrainingOptions) -> Self {

        if data.get_data().is_empty() {
            let models : Vec<ClmModel> = (0..options.ensemble_size).map(|_| ClmModel::from_buffer(vec![])).collect();
            return EnsembleModel { models: models };
        }

        let chunks  =
            Box::new(data.split_into_chunks(options.ensemble_size));
        

        let progress_bar = indicatif::ProgressBar::new(options.ensemble_size as u64);


        // train a model for each chunk
        let models = chunks.par_bridge().map(|chunk| {
            let model = train_model(chunk.get_data(), options);
            progress_bar.inc(1);
            model
        }).collect();

        progress_bar.finish();

        EnsembleModel { models }
    }

    pub fn save_checkpoint(&self, path: &str) {
        // write all models to a sqlite database
        let conn = Connection::open(path).unwrap();
        conn.execute("CREATE TABLE models (id INTEGER PRIMARY KEY, model BLOB)", []).unwrap();

        for (i, model) in self.models.iter().enumerate() {
            let mut stmt = conn.prepare("INSERT INTO models (id, model) VALUES (?, ?)").unwrap();
            let model_bytes = model.to_buffer();
            stmt.execute((&i, &model_bytes)).unwrap();
        }

        conn.close().unwrap();
    }

    pub fn from_checkpoint(path: &str) -> Self {
        // read all models from a sqlite database
        let  conn = Connection::open(path).unwrap();
        let models;
        {
            let mut stmt = conn.prepare("SELECT model FROM models").unwrap();
            models = stmt.query_map([], |row| {
                let model_bytes: Vec<u8> = row.get(0)?;
                Ok(ClmModel::from_buffer(model_bytes))
            }).unwrap().map(|x| x.unwrap()).collect();
        }
        conn.close().unwrap();
        EnsembleModel { models }
    }

    pub fn compressed_size(&self, tokens: &Vec<Token>) -> f64 {
        let mut total_size = 0.0;
        for model in &self.models {
            total_size += model.compress(tokens).len() as f64;
        }
        total_size / self.models.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::dataset::Dataset;
    use crate::backend::ensemble_model::EnsembleModel;
    use crate::backend::training_options::TrainingOptions;

    #[test]
    fn training_works() {
        let dataset = Dataset::test_dataset();

        let mut options = TrainingOptions::default();
        options.ensemble_size = 2;
        let trained_model = EnsembleModel::train(dataset, &options);

        assert_eq!(trained_model.models.len(), 2);
    }

    #[test]
    fn compressed_size_improves() {
        let dataset = Dataset::test_dataset();

        let mut options = TrainingOptions::default();
        options.ensemble_size = 2;
        let trained_model = EnsembleModel::train(dataset.clone(), &options);
        let untrained_model = EnsembleModel::train(Dataset::empty(), &options);

        let trained_size = trained_model.compressed_size(&dataset.get_data()[0]);
        let naive_size = untrained_model.compressed_size(&dataset.get_data()[0]);

        println!("Trained size: {}", trained_size);
        println!("Naive size: {}", naive_size);
        assert!(trained_size < naive_size);
    }

    #[test]
    fn save_and_load_ensemble_model() {
        let dataset = Dataset::test_dataset();

        let mut options = TrainingOptions::default();
        options.ensemble_size = 2;
        let trained_model = EnsembleModel::train(dataset, &options);

        trained_model.save_checkpoint("test.ensemble");

        let loaded_model = EnsembleModel::from_checkpoint("test.ensemble");

        assert_eq!(trained_model.models.len(), loaded_model.models.len());
        assert_eq!(trained_model.models[0].get_dictionary_size(), loaded_model.models[0].get_dictionary_size());

        // clean up
        std::fs::remove_file("test.ensemble").unwrap();
    }
}

