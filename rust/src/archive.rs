//! Model archive loader — reads TFLite models and weights from a ZIP file.
//!
//! The archive uses canonical names so the same binary works with different
//! target-specific model variants (Neutron, standard INT8, etc.):
//!
//! ```text
//! encoder.tflite              — Image encoder (INT8)
//! attention.tflite            — Transformer attention (FP16 or FP32)
//! heads_a.tflite              — Decoder heads Part A (INT8)
//! heads_b.tflite              — Decoder heads Part B (INT8)
//! tokens.tflite               — Token MLP (dynamic range)
//! prompt_encoder.safetensors  — Prompt encoder weights
//! ```

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Loaded model archive contents (raw bytes, not yet parsed).
pub struct ModelArchive {
    entries: HashMap<String, Vec<u8>>,
}

/// Expected canonical model names.
const REQUIRED_ENTRIES: &[&str] = &[
    "encoder.tflite",
    "attention.tflite",
    "heads_a.tflite",
    "heads_b.tflite",
    "tokens.tflite",
    "prompt_encoder.safetensors",
];

impl ModelArchive {
    /// Load all model files from a ZIP archive into memory.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open archive '{}': {}", path.display(), e))?;

        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| format!("Invalid ZIP archive '{}': {}", path.display(), e))?;

        let mut entries = HashMap::new();

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            let name = entry.name().to_string();

            // Skip directories and hidden files
            if name.ends_with('/') || name.starts_with('.') {
                continue;
            }

            // Use just the filename (strip any directory prefix)
            let filename = Path::new(&name)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or(name.clone());

            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;

            entries.insert(filename, buf);
        }

        // Validate all required entries are present
        let mut missing = Vec::new();
        for &name in REQUIRED_ENTRIES {
            if !entries.contains_key(name) {
                missing.push(name);
            }
        }
        if !missing.is_empty() {
            return Err(format!(
                "Archive '{}' is missing required models: {}",
                path.display(),
                missing.join(", ")
            )
            .into());
        }

        Ok(Self { entries })
    }

    /// Take ownership of a model's bytes (removes it from the archive).
    ///
    /// This is used with `Model::from_bytes()` which takes ownership of the buffer.
    pub fn take(&mut self, name: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        self.entries
            .remove(name)
            .ok_or_else(|| format!("Model '{}' not found in archive", name).into())
    }

    /// Borrow a model's bytes (for safetensors which deserializes from a slice).
    pub fn get(&self, name: &str) -> Result<&[u8], Box<dyn std::error::Error>> {
        self.entries
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| format!("Entry '{}' not found in archive", name).into())
    }

    /// List all entries with sizes.
    pub fn list(&self) -> Vec<(&str, usize)> {
        let mut items: Vec<_> = self
            .entries
            .iter()
            .map(|(k, v)| (k.as_str(), v.len()))
            .collect();
        items.sort_by_key(|(name, _)| *name);
        items
    }
}
