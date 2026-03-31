//! TFLite model runner with quantization helpers.
//!
//! Wraps `edgefirst_tflite::Interpreter` with convenience methods for
//! setting inputs, invoking inference, and reading outputs. Supports
//! both file-based and buffer-based model loading.

use std::time::{Duration, Instant};

use edgefirst_tflite::{Delegate, Interpreter, Library, Model, TensorType};

/// Wrapper around an `edgefirst_tflite::Interpreter` providing convenience
/// methods for setting inputs, invoking inference, and reading outputs.
pub struct TfLiteRunner<'a> {
    interpreter: Interpreter<'a>,
}

impl<'a> TfLiteRunner<'a> {
    /// Create a new runner from a pre-loaded model with optional delegates.
    ///
    /// `delegate_paths` are tried in order; the first that loads successfully
    /// is used. If none succeed, runs on CPU.
    pub fn new(
        lib: &'a Library,
        model: &'a Model<'a>,
        threads: usize,
        delegate_paths: &[&str],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut builder = Interpreter::builder(lib)?.num_threads(threads as i32);

        for path in delegate_paths {
            match Delegate::load(path) {
                Ok(delegate) => {
                    builder = builder.delegate(delegate);
                    break;
                }
                Err(e) => {
                    eprintln!("  Delegate '{}' failed to load: {}", path, e);
                }
            }
        }

        let interpreter = builder.build(model)?;
        Ok(Self { interpreter })
    }

    /// Create a runner with the built-in XNNPACK delegate for multi-threaded
    /// FP16 CPU acceleration (ARMv8.2 NEON).
    pub fn with_xnnpack(
        lib: &'a Library,
        model: &'a Model<'a>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let delegate = Delegate::xnnpack(lib, threads as i32)?;
        let interpreter = Interpreter::builder(lib)?
            .num_threads(threads as i32)
            .delegate(delegate)
            .build(model)?;
        Ok(Self { interpreter })
    }

    // --- Input setters ---

    /// Set FP32 input data at the given tensor index.
    pub fn set_input_f32_at(
        &mut self,
        index: usize,
        data: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut inputs = self.interpreter.inputs_mut()?;
        let tensor = &mut inputs[index];
        tensor.copy_from_slice(data)?;
        Ok(())
    }

    /// Set FP32 input data at tensor index 0.
    pub fn set_input_f32(&mut self, data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        self.set_input_f32_at(0, data)
    }

    /// Set INT8 input by quantizing from FP32 values, at the given index.
    pub fn set_input_quantized_at(
        &mut self,
        index: usize,
        data: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut inputs = self.interpreter.inputs_mut()?;
        let tensor = &mut inputs[index];
        let params = tensor.quantization_params();
        let scale = params.scale;
        let zp = params.zero_point;
        let slice = tensor.as_mut_slice::<i8>()?;
        for (dst, &src) in slice.iter_mut().zip(data.iter()) {
            *dst = (src / scale + zp as f32).round().clamp(-128.0, 127.0) as i8;
        }
        Ok(())
    }

    /// Set INT8 input by quantizing from FP32 values at tensor index 0.
    pub fn set_input_quantized(
        &mut self,
        data: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.set_input_quantized_at(0, data)
    }

    /// Set pre-quantized INT8 input directly at tensor index 0.
    pub fn set_input_i8(&mut self, data: &[i8]) -> Result<(), Box<dyn std::error::Error>> {
        let mut inputs = self.interpreter.inputs_mut()?;
        let tensor = &mut inputs[0];
        tensor.copy_from_slice(data)?;
        Ok(())
    }

    // --- Inference ---

    /// Invoke inference and return the elapsed wall-clock time.
    pub fn invoke(&mut self) -> Result<Duration, Box<dyn std::error::Error>> {
        let t = Instant::now();
        self.interpreter.invoke()?;
        Ok(t.elapsed())
    }

    // --- Output readers ---

    /// Get FP32 output at the given tensor index.
    pub fn output_f32_at(
        &self,
        index: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let outputs = self.interpreter.outputs()?;
        Ok(outputs[index].as_slice::<f32>()?.to_vec())
    }

    /// Get FP32 output at tensor index 0.
    pub fn output_f32(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.output_f32_at(0)
    }

    /// Get output dequantized from INT8 to FP32 at the given tensor index.
    pub fn output_dequantized_at(
        &self,
        index: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let outputs = self.interpreter.outputs()?;
        let tensor = &outputs[index];
        let params = tensor.quantization_params();
        let scale = params.scale;
        let zp = params.zero_point;
        let slice = tensor.as_slice::<i8>()?;
        Ok(slice
            .iter()
            .map(|&v| (v as f32 - zp as f32) * scale)
            .collect())
    }

    /// Get output dequantized from INT8 to FP32 at tensor index 0.
    pub fn output_dequantized(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.output_dequantized_at(0)
    }

    /// Get all outputs as named f32 vectors with shapes, auto-dequantizing INT8.
    pub fn all_outputs_f32(
        &self,
    ) -> Result<Vec<(String, Vec<f32>, Vec<usize>)>, Box<dyn std::error::Error>> {
        let outputs = self.interpreter.outputs()?;
        let mut result = Vec::new();
        for out in outputs.iter() {
            let name = out.name().to_string();
            let shape = out.shape()?;
            let data = if out.tensor_type() == TensorType::Int8 {
                let params = out.quantization_params();
                out.as_slice::<i8>()?
                    .iter()
                    .map(|&v| (v as f32 - params.zero_point as f32) * params.scale)
                    .collect()
            } else {
                out.as_slice::<f32>()?.to_vec()
            };
            result.push((name, data, shape));
        }
        Ok(result)
    }

    /// Return the number of input tensors.
    pub fn input_count(&self) -> usize {
        self.interpreter.input_count()
    }

    /// Return the number of output tensors.
    pub fn output_count(&self) -> usize {
        self.interpreter.output_count()
    }
}
