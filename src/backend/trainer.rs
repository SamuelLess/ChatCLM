use std::ffi::{c_uint, c_void};
use itertools::Itertools;
use zstd_sys::{ZDICT_isError, ZDICT_optimizeTrainFromBuffer_fastCover};
use crate::backend::{BYTES_PER_TOKEN, Token, tokens_to_bytes};
use crate::backend::clm_model::ClmModel;
use crate::backend::training_options::TrainingOptions;

pub fn train_model<'a>(input_tokens: Vec<Vec<Token>>, training_options: TrainingOptions) -> ClmModel<'a> {
    
    if input_tokens.is_empty() {
        return ClmModel::from_buffer(vec![]);
    }
    
    let raw_data = input_tokens.iter().flat_map(|x| tokens_to_bytes(x)).collect_vec();
    let sizes = input_tokens.iter().map(|x| x.len() * BYTES_PER_TOKEN).collect_vec();
    let buffer_size = raw_data.len();
    assert_eq!(sizes.iter().sum::<usize>(), raw_data.len(), "Sizes sum doesn't match raw data size");
    let mut buffer = vec![0u8; buffer_size];
    let mut parameters = training_options.to_zdict_params();
    let mut size = 0;
    unsafe {
        size = ZDICT_optimizeTrainFromBuffer_fastCover(
            buffer.as_mut_ptr() as *mut c_void,
            buffer_size,
            raw_data.as_ptr() as *mut c_void,
            sizes.as_ptr(),
            sizes.len() as c_uint,
            &mut parameters,
        );
        println!("Selected parameters: {:?}", parameters);
        println!("Dictionary size: {:?}", size);

        if ZDICT_isError(size) != 0 {
            panic!("Failed to train dictionary");
        }
    }

    println!("Dictionary trained, resizing buffer to size: {}", size);
    buffer.resize(size, 0);
    println!("Buffer resized {}", buffer.len());

    
    ClmModel::from_buffer(buffer)
}