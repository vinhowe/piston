#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use ndarray::Array4;
    use ratchet::{shape, DType, Device, Tensor};
    use serde::Deserialize;
    use serde_json::Value;
    use wasm_bindgen_test::*;

    use crate::test_utils::log_init;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    const TEST_SUITE_JSON: &str = include_str!("../test_suite.json");

    /// Represents a single test case in the JSON.
    #[derive(Debug, Deserialize)]
    struct TestCase {
        // /// Human-readable name (e.g. "addition").
        // name: String,
        /// The name of the operation (e.g. "add", "matmul", "conv2d", "relu", etc.).
        operation: String,
        /// Inputs are stored as arbitrary JSON. We'll parse them depending on the operation.
        inputs: Value,
        /// The expected output for this test. Also arbitrary JSON to parse as needed.
        output: Value,
    }

    /// Converts a nested JSON array of shape (m x n) into an `ndarray::Array2<f32>`.
    fn json_to_array2_f32(value: &Value, device: &Device) -> Tensor {
        // We expect a 2D list structure, e.g. [[0.1, 0.2], [0.3, 0.4]]
        // Adjust the parsing logic if you have more dimensions or a different data shape.
        let rows = value.as_array().expect("Expected outer array");

        // Collect each row as a Vec<f32>
        let row_vecs: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("Expected inner array")
                    .iter()
                    .map(|num| num.as_f64().expect("Expected a numeric value") as f32)
                    .collect()
            })
            .collect();

        // Get the shape
        let nrows = row_vecs.len();
        let ncols = row_vecs[0].len(); // assume at least one row

        // Flatten the data
        let flattened: Vec<f32> = row_vecs.into_iter().flatten().collect();

        // Create tensor directly from data
        Tensor::from_data(flattened, shape![nrows, ncols], device.clone())
    }

    /// Parse a 4D JSON array into an `Array4<f32>`.
    /// Expects something like:
    /// [
    ///   [
    ///     [ [0.1, 0.2], [0.3, 0.4] ],
    ///     [ [0.5, 0.6], [0.7, 0.8] ]
    ///   ],
    ///   [
    ///     [ [1.1, 1.2], [1.3, 1.4] ],
    ///     [ [1.5, 1.6], [1.7, 1.8] ]
    ///   ]
    /// ]
    /// which corresponds to shape (batch_size, channels, height, width).
    pub fn json_to_array4_f32(value: &Value) -> Array4<f32> {
        // Top-level array is "batch".
        let batch = value
            .as_array()
            .expect("Expected top-level JSON array for 4D data.");

        // In many deep-learning use cases, the shape is [N, C, H, W].
        // We'll parse each dimension carefully.
        // Collect all data in a flat Vec<f32> so we can feed it to `Array4::from_shape_vec`.
        let mut data = Vec::new();
        let mut batch_size = 0_usize;
        let mut channel_size = 0_usize;
        let mut height_size = 0_usize;
        let mut width_size = 0_usize;

        for channels in batch {
            // Each "channels" is itself an array.
            let channels_arr = channels
                .as_array()
                .expect("Expected array for the second dimension.");

            batch_size += 1;
            // If we haven’t set channel_size yet, set it now.
            if channel_size == 0 {
                channel_size = channels_arr.len();
            }

            for height in channels_arr {
                let height_arr = height
                    .as_array()
                    .expect("Expected array for the third dimension.");
                if height_size == 0 {
                    height_size = height_arr.len();
                }

                for width in height_arr {
                    let width_arr = width
                        .as_array()
                        .expect("Expected array for the fourth dimension.");
                    if width_size == 0 {
                        width_size = width_arr.len();
                    }

                    // Finally, read numeric values.
                    for val in width_arr {
                        data.push(val.as_f64().expect("Expected float in JSON.") as f32);
                    }
                }
            }
        }

        Array4::from_shape_vec((batch_size, channel_size, height_size, width_size), data)
            .expect("Failed to create Array4 from parsed JSON data.")
    }

    fn parse_scalar_f32(value: &Value) -> f32 {
        value.as_f64().expect("Expected a JSON float for powf") as f32
    }

    async fn run_op_test_case(test_case: &TestCase, device: &Device) -> anyhow::Result<()> {
        fn parse_inputs_2d(
            inputs: &Value,
            key1: &str,
            key2: &str,
            device: &Device,
        ) -> (Tensor, Tensor) {
            let arr1 = json_to_array2_f32(&inputs[key1], device);
            let arr2 = json_to_array2_f32(&inputs[key2], device);
            (arr1, arr2)
        }

        let op = test_case.operation.as_str();
        log::info!("op = {:?}", op);
        match op {
            "add" => {
                // Parse inputs
                let (x, y) = parse_inputs_2d(&test_case.inputs, "x", "y", device);
                // Parse expected output
                let expected = json_to_array2_f32(&test_case.output, device)
                    .to(&Device::CPU)
                    .await?;

                // Perform operation
                let result = (x + y).unwrap().to(&Device::CPU).await.unwrap();

                // Compare
                result.all_close(&expected, 1e-5, 1e-5)?;
            }

            "matmul" => {
                let (x, y) = parse_inputs_2d(&test_case.inputs, "x", "y", device);
                let expected = json_to_array2_f32(&test_case.output, device)
                    .to(&Device::CPU)
                    .await?;

                let result = x.matmul(y, false, false)?.to(&Device::CPU).await?;

                result.all_close(&expected, 1e-5, 1e-5)?;
            }

            "gather" => {
                let (x, ids) = parse_inputs_2d(&test_case.inputs, "input", "index", device);
                // Cast ids to i32
                let ids = ids.cast(DType::I32)?;
                let expected = json_to_array2_f32(&test_case.output, device)
                    .to(&Device::CPU)
                    .await?;

                let result = x.gather(ids, 1)?.to(&Device::CPU).await?;

                result.all_close(&expected, 1e-5, 1e-5)?;
            }

            "scatter_add" => {
                let (x, ids) = parse_inputs_2d(&test_case.inputs, "input", "index", device);
                // Cast ids to i32
                let ids = ids.cast(DType::I32)?;
                let expected = json_to_array2_f32(&test_case.output, device)
                    .to(&Device::CPU)
                    .await?;

                let dst = Tensor::zeros::<f32>(x.shape(), device);

                let result = dst
                    .scatter_add(ids.clone(), x.clone(), 1)?
                    .to(&Device::CPU)
                    .await?;

                log::warn!("ids = {:?}", ids.to(&Device::CPU).await?);
                log::warn!("result = {:?}", result);
                log::warn!("expected = {:?}", expected);
                log::warn!("x = {:?}", x.to(&Device::CPU).await?);

                result.all_close(&expected, 1e-5, 1e-5)?;
            }

            "powf" => {
                let x = json_to_array2_f32(&test_case.inputs["x"], device);
                let y = parse_scalar_f32(&test_case.inputs["y"]);
                let expected = json_to_array2_f32(&test_case.output, device)
                    .to(&Device::CPU)
                    .await?;

                let result = x.powf(y)?.to(&Device::CPU).await?;

                result.all_close(&expected, 1e-5, 1e-5)?;
            }

            // "conv2d" => {
            //     // Example for 4D convolution data: parse (batch, in_channels, height, width)
            //     // plus kernel. Let’s assume your JSON structure is something like:
            //     //  "inputs": {
            //     //      "input": [[[...], [...], ... ], ... ],
            //     //      "weight": [[[...], ...], ... ],
            //     //      "stride": 1,
            //     //      "padding": 0
            //     //  }

            //     let input_4d = json_to_array4_f32(&test_case.inputs["input"]);
            //     let weight_4d = json_to_array4_f32(&test_case.inputs["weight"]);
            //     let stride = test_case.inputs["stride"].as_u64().unwrap() as usize;
            //     let padding = test_case.inputs["padding"].as_u64().unwrap() as usize;

            //     let expected_4d = json_to_array4_f32(&test_case.output);

            //     // (You’d implement conv2d in Rust or use a library—skipped for brevity.)
            //     let result_4d = rust_conv2d(&input_4d, &weight_4d, stride, padding);

            //     // For 4D arrays, you'd have a compare function that handles 4D data
            //     compare_and_log_4d(&test_case.name, op, &result_4d, &expected_4d, 1e-5);
            // }
            "relu" => {
                let x = json_to_array2_f32(&test_case.inputs, device);
                let expected = json_to_array2_f32(&test_case.output, device)
                    .to(&Device::CPU)
                    .await?;

                let result = x.relu()?.to(&Device::CPU).await?;

                result.all_close(&expected, 1e-5, 1e-5)?;
            }
            _ => {
                println!("Operation '{}' not yet implemented.", op);
            }
        }

        Ok(())
    }

    async fn run_test_cases(test_cases: &[TestCase]) -> anyhow::Result<()> {
        let device = Device::request_device(ratchet::DeviceRequest::GPU)
            .await
            .unwrap();

        for test_case in test_cases {
            run_op_test_case(test_case, &device).await?;
        }
        Ok(())
    }

    #[wasm_bindgen_test]
    async fn test_run_test_cases() {
        log_init();
        log::error!("test_run_test_cases");
        let test_cases: Vec<TestCase> = serde_json::from_str(TEST_SUITE_JSON).unwrap();
        log::error!("test_run_test_cases 2");
        run_test_cases(&test_cases).await.unwrap();
    }
}
