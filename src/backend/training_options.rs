use std::ffi::c_int;

pub struct TrainingOptions {
    pub d: u32,
    pub f: u32,
    pub k: u32,
    pub steps: u32,
    pub nb_threads: u32,
    pub split_point: f64,
    pub accel: u32,
    pub shrink_dict: u32,
    pub shrink_dict_max_regression: u32,
    pub compression_level: u32,
    pub dictionary_size_percentage: f64 /* 0.0 to 1.0, how big the dictionary should be compared to the input data */,
    pub ensemble_size: usize, /* number of models to train */
}

impl TrainingOptions {
    pub fn new() -> Self {
        TrainingOptions {
            d: 8,
            f: 25,
            k: 50,
            steps: 4,
            nb_threads: 8,
            split_point: 0.0,
            accel: 1,
            shrink_dict: 0,
            shrink_dict_max_regression: 0,
            compression_level: 3,
            dictionary_size_percentage: 1.0,
            ensemble_size: 1,
        }
    }

    pub fn to_zdict_params(&self) -> zstd_sys::ZDICT_fastCover_params_t {
        zstd_sys::ZDICT_fastCover_params_t {
            k: self.k,
            d: self.d,
            f: self.f,
            steps: self.steps,
            nbThreads: self.nb_threads,
            splitPoint: self.split_point,
            accel: self.accel,
            shrinkDict: self.shrink_dict,
            shrinkDictMaxRegression: self.shrink_dict_max_regression,
            zParams: zstd_sys::ZDICT_params_t {
                compressionLevel: self.compression_level as c_int,
                notificationLevel: 2,
                dictID: 0,
            },
        }
    }

    pub fn default() -> Self {
        TrainingOptions::new()
    }
}
