use std::path::Path;

use safetensors::{Dtype, SafeTensors};

// ---------------------------------------------------------------------------
// Float trait
// ---------------------------------------------------------------------------

/// Numeric trait abstracting over f32 (and future f16) storage for embeddings.
pub trait Float: Copy + Default + std::fmt::Debug {
    const PI: Self;
    fn from_f32(v: f32) -> Self;
    fn from_f16(v: half::f16) -> Self;
    fn to_f32(self) -> f32;
    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
}

impl Float for f32 {
    const PI: Self = std::f32::consts::PI;

    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }

    #[inline]
    fn from_f16(v: half::f16) -> Self {
        v.to_f32()
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn sin(self) -> Self {
        f32::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        f32::cos(self)
    }
}

// ---------------------------------------------------------------------------
// PromptEncoder
// ---------------------------------------------------------------------------

/// Loaded weights for the SAM prompt encoder (point / mask branch).
///
/// All embedding tables are stored in the generic float type `T`, which is
/// currently always `f32`.  An `f16` implementation can be added later once
/// the NPU path requires it.
///
/// Layout:
/// - `gaussian_matrix`   — shape (2, 128) flattened row-major, 256 elements
/// - `point_embeddings`  — 4 × 256 elements, one per label (0-3)
/// - `not_a_point_embed` — 256 elements, used for label -1 (padding)
/// - `no_mask_embed`     — 256 elements, used when no mask prompt is given
/// - `output_tokens`     — shape (5, 256) flattened row-major, 1280 elements
///                         (iou_token + 4 mask_tokens)
pub struct PromptEncoder<T: Float> {
    pub gaussian_matrix: [T; 256],
    pub point_embeddings: [[T; 256]; 4],
    pub not_a_point_embed: [T; 256],
    pub no_mask_embed: [T; 256],
    pub output_tokens: [T; 1280],
}

impl<T: Float> PromptEncoder<T> {
    /// Load weights from a safetensors file at `path`.
    ///
    /// Tensors may be stored as either F32 or F16; each element is converted
    /// to `T` via the appropriate `Float` constructor.
    pub fn from_safetensors(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        Self::from_safetensors_bytes(&bytes)
    }

    /// Load weights from a safetensors byte buffer (e.g. from a ZIP archive).
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let tensors = SafeTensors::deserialize(bytes)?;

        let gaussian_matrix =
            to_array(&load_tensor::<T>(&tensors, "gaussian_matrix", 256)?);

        let point_embeddings = [
            to_array(&load_tensor::<T>(&tensors, "point_embeddings.0", 256)?),
            to_array(&load_tensor::<T>(&tensors, "point_embeddings.1", 256)?),
            to_array(&load_tensor::<T>(&tensors, "point_embeddings.2", 256)?),
            to_array(&load_tensor::<T>(&tensors, "point_embeddings.3", 256)?),
        ];

        let not_a_point_embed =
            to_array(&load_tensor::<T>(&tensors, "not_a_point_embed", 256)?);
        let no_mask_embed =
            to_array(&load_tensor::<T>(&tensors, "no_mask_embed", 256)?);
        let output_tokens =
            to_array(&load_tensor::<T>(&tensors, "output_tokens", 1280)?);

        Ok(Self {
            gaussian_matrix,
            point_embeddings,
            not_a_point_embed,
            no_mask_embed,
            output_tokens,
        })
    }

    /// Encode point prompts and image embeddings into sparse and dense tokens
    /// suitable for the SAM decoder.
    ///
    /// # Arguments
    /// - `point_coords` — flat (1, N, 2) array, length 2*N, pixel coordinates
    /// - `point_labels` — flat (1, N) array, length N; 0/1/2/3 or -1 for padding
    /// - `image_embeddings` — flat (1, 256, 64, 64) NCHW array, 1 048 576 elements
    /// - `has_mask_input` — must be 0.0; mask refinement is not yet implemented
    ///
    /// Returns `(sparse_embeddings, image_src)` where:
    /// - `sparse_embeddings` has shape (5+N, 256) flattened
    /// - `image_src` has shape (256, 64, 64) = 1 048 576 elements
    pub fn encode(
        &self,
        point_coords: &[f32],
        point_labels: &[f32],
        image_embeddings: &[T],
        has_mask_input: f32,
    ) -> (Vec<T>, Vec<T>) {
        assert!(
            has_mask_input == 0.0,
            "Mask refinement is not implemented; has_mask_input must be 0.0"
        );
        let n = point_labels.len();
        assert_eq!(
            point_coords.len(),
            n * 2,
            "point_coords must have 2*N elements for N points"
        );

        let sparse = self.encode_sparse(point_coords, point_labels, n);
        let image_src = self.encode_dense_no_mask(image_embeddings);
        (sparse, image_src)
    }

    /// Compute sparse embeddings: Gaussian positional encoding for each point,
    /// add label embeddings, then prepend the 5 output tokens.
    ///
    /// Returns a flat Vec<T> of length (5 + N) * 256.
    fn encode_sparse(&self, point_coords: &[f32], point_labels: &[f32], n: usize) -> Vec<T> {
        let total_tokens = 5 + n;
        let mut out = vec![T::default(); total_tokens * 256];

        // Copy output_tokens (5 rows × 256) into the first 1280 elements.
        out[..1280].copy_from_slice(&self.output_tokens);

        // Encode each point.
        for i in 0..n {
            let x = point_coords[i * 2];
            let y = point_coords[i * 2 + 1];

            // Normalize pixel coordinates to [-1, 1] (image size 1024).
            let nx = (x + 0.5) / 1024.0 * 2.0 - 1.0;
            let ny = (y + 0.5) / 1024.0 * 2.0 - 1.0;

            let row_offset = (5 + i) * 256;

            // Gaussian positional encoding: 256-dim vector (128 sin, 128 cos).
            for j in 0..128 {
                let val = nx * self.gaussian_matrix[j].to_f32()
                    + ny * self.gaussian_matrix[128 + j].to_f32();
                let angle = 2.0 * std::f32::consts::PI * val;
                out[row_offset + j] = T::from_f32(angle.sin());
                out[row_offset + 128 + j] = T::from_f32(angle.cos());
            }

            // Add label embedding.
            let label = point_labels[i];
            let label_embed: &[T; 256] = if label < 0.0 {
                &self.not_a_point_embed
            } else {
                let idx = label as usize;
                assert!(idx < 4, "point label must be 0-3 or -1, got {}", idx);
                &self.point_embeddings[idx]
            };

            for d in 0..256 {
                out[row_offset + d] = out[row_offset + d].add(label_embed[d]);
            }
        }

        out
    }

    /// Compute dense embeddings by broadcasting `no_mask_embed` over the
    /// spatial dimensions of the image embeddings.
    ///
    /// Input layout: (1, 256, 64, 64) NCHW, i.e. channel c occupies
    /// elements `[c*4096 .. (c+1)*4096]`.  The batch dimension is dropped in
    /// the output (shape 256 × 64 × 64 = 1 048 576 elements).
    fn encode_dense_no_mask(&self, image_embeddings: &[T]) -> Vec<T> {
        assert_eq!(
            image_embeddings.len(),
            1_048_576,
            "image_embeddings must have 1 048 576 elements (1×256×64×64)"
        );

        let mut out = vec![T::default(); 1_048_576];
        for c in 0..256 {
            let bias = self.no_mask_embed[c];
            let base = c * 4096;
            for s in 0..4096 {
                out[base + s] = image_embeddings[base + s].add(bias);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a named tensor from `tensors`, converting each element to `T`.
///
/// Returns an error if the tensor is missing, has an unexpected dtype, or
/// does not contain exactly `expected_len` elements.
fn load_tensor<T: Float>(
    tensors: &SafeTensors,
    name: &str,
    expected_len: usize,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let view = tensors
        .tensor(name)
        .map_err(|e| format!("tensor '{}' not found: {}", name, e))?;

    let data = view.data();
    let dtype = view.dtype();

    let values: Vec<T> = match dtype {
        Dtype::F32 => {
            if data.len() != expected_len * 4 {
                return Err(format!(
                    "tensor '{}': expected {} f32 elements ({} bytes), got {} bytes",
                    name,
                    expected_len,
                    expected_len * 4,
                    data.len()
                )
                .into());
            }
            data.chunks_exact(4)
                .map(|b| {
                    let arr: [u8; 4] = b.try_into().unwrap();
                    T::from_f32(f32::from_le_bytes(arr))
                })
                .collect()
        }
        Dtype::F16 => {
            if data.len() != expected_len * 2 {
                return Err(format!(
                    "tensor '{}': expected {} f16 elements ({} bytes), got {} bytes",
                    name,
                    expected_len,
                    expected_len * 2,
                    data.len()
                )
                .into());
            }
            data.chunks_exact(2)
                .map(|b| {
                    let arr: [u8; 2] = b.try_into().unwrap();
                    T::from_f16(half::f16::from_le_bytes(arr))
                })
                .collect()
        }
        other => {
            return Err(
                format!("tensor '{}': unsupported dtype {:?}", name, other).into(),
            )
        }
    };

    Ok(values)
}

/// Copy a slice into a fixed-size array.
///
/// Panics if `slice.len() != N`.
fn to_array<T: Float + Copy, const N: usize>(slice: &[T]) -> [T; N] {
    assert_eq!(
        slice.len(),
        N,
        "expected {} elements, got {}",
        N,
        slice.len()
    );
    let mut arr = [T::default(); N];
    arr.copy_from_slice(slice);
    arr
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Float trait basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_float_f32_basic_ops() {
        let a = 1.0_f32;
        let b = 2.0_f32;
        assert_eq!(a.add(b), 3.0_f32);
        assert_eq!(a.mul(b), 2.0_f32);

        let half_pi = f32::PI / 2.0;
        let sin_val = half_pi.sin();
        assert!((sin_val - 1.0).abs() < 1e-6, "sin(π/2) should be ≈1, got {}", sin_val);

        let cos_val = 0.0_f32.cos();
        assert!((cos_val - 1.0).abs() < 1e-6, "cos(0) should be ≈1, got {}", cos_val);
    }

    #[test]
    fn test_float_f32_f16_conversion() {
        let original: f32 = 1.5;
        let as_f16 = half::f16::from_f32(original);
        let back: f32 = f32::from_f16(as_f16);
        assert!(
            (back - 1.5).abs() < 1e-3,
            "round-trip f32→f16→f32 of 1.5 should be 1.5, got {}",
            back
        );
    }

    // -----------------------------------------------------------------------
    // Helper: to_array
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_array() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let arr: [f32; 4] = to_array(&v);
        assert_eq!(arr, [1.0_f32, 2.0, 3.0, 4.0]);
    }

    // -----------------------------------------------------------------------
    // Helpers to build a zeroed PromptEncoder quickly
    // -----------------------------------------------------------------------

    fn zeroed_encoder() -> PromptEncoder<f32> {
        PromptEncoder {
            gaussian_matrix: [0.0_f32; 256],
            point_embeddings: [[0.0_f32; 256]; 4],
            not_a_point_embed: [0.0_f32; 256],
            no_mask_embed: [0.0_f32; 256],
            output_tokens: [0.0_f32; 1280],
        }
    }

    // -----------------------------------------------------------------------
    // encode_sparse
    // -----------------------------------------------------------------------

    /// With a zero gaussian_matrix and zero embeddings, encoding a single
    /// foreground point at (512, 512) should produce:
    ///   - 6 tokens total (5 output_tokens + 1 point)
    ///   - first 5 rows are all zero
    ///   - point row: sin(0)=0 for dims 0..128, cos(0)=1 for dims 128..256
    #[test]
    fn test_encode_sparse_single_point() {
        let enc = zeroed_encoder();

        let point_coords = vec![512.0_f32, 512.0_f32];
        let point_labels = vec![1.0_f32]; // foreground

        let result = enc.encode_sparse(&point_coords, &point_labels, 1);

        assert_eq!(result.len(), 6 * 256, "should have 6 token rows");

        // First 5 rows (output_tokens) must all be zero.
        for (i, &v) in result[..5 * 256].iter().enumerate() {
            assert_eq!(v, 0.0_f32, "output_token element {} should be 0", i);
        }

        let point_row = &result[5 * 256..6 * 256];

        // Gaussian matrix is zero → val = 0 → angle = 0 → sin(0)=0, cos(0)=1.
        // Label embedding (point_embeddings[1]) is also zero, so no shift.
        for j in 0..128 {
            assert!(
                (point_row[j] - 0.0).abs() < 1e-6,
                "sin dim {} should be 0, got {}",
                j,
                point_row[j]
            );
        }
        for j in 128..256 {
            assert!(
                (point_row[j] - 1.0).abs() < 1e-6,
                "cos dim {} should be 1, got {}",
                j,
                point_row[j]
            );
        }
    }

    // -----------------------------------------------------------------------
    // encode_dense_no_mask
    // -----------------------------------------------------------------------

    /// no_mask_embed[0]=0.5, no_mask_embed[1]=-0.3, rest=0.
    /// image_embeddings = all 1.0.
    /// Expected: channel 0 → 1.5, channel 1 → 0.7, channels 2..255 → 1.0.
    #[test]
    fn test_encode_dense_no_mask() {
        let mut enc = zeroed_encoder();
        enc.no_mask_embed[0] = 0.5_f32;
        enc.no_mask_embed[1] = -0.3_f32;

        let image = vec![1.0_f32; 1_048_576];
        let result = enc.encode_dense_no_mask(&image);

        assert_eq!(result.len(), 1_048_576);

        // Channel 0: 1.0 + 0.5 = 1.5
        for s in 0..4096 {
            assert!(
                (result[s] - 1.5).abs() < 1e-6,
                "channel 0 pixel {} should be 1.5, got {}",
                s,
                result[s]
            );
        }

        // Channel 1: 1.0 + (-0.3) = 0.7
        for s in 0..4096 {
            assert!(
                (result[4096 + s] - 0.7).abs() < 1e-6,
                "channel 1 pixel {} should be 0.7, got {}",
                s,
                result[4096 + s]
            );
        }

        // Channels 2..255: 1.0 + 0.0 = 1.0
        for c in 2..256 {
            for s in 0..4096 {
                assert!(
                    (result[c * 4096 + s] - 1.0).abs() < 1e-6,
                    "channel {} pixel {} should be 1.0, got {}",
                    c,
                    s,
                    result[c * 4096 + s]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // encode — mask input guard
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "Mask refinement")]
    fn test_encode_rejects_mask_input() {
        let enc = zeroed_encoder();
        let image = vec![0.0_f32; 1_048_576];
        enc.encode(&[512.0, 512.0], &[1.0], &image, 1.0);
    }
}
