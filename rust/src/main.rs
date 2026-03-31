//! NanoSAM CLI — segment images using NanoSAM with optional NPU acceleration.

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use edgefirst_image::ImageProcessorTrait as _;
use nanosam::{BBox, Mask, Point, PointLabel, Prompt, SegmentationResult, SessionConfig};

#[derive(Parser)]
#[command(
    name = "nanosam",
    about = "NanoSAM: Real-time Segment Anything on edge devices"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Segment an image with a box or point prompt
    Segment {
        /// Path to model archive ZIP
        #[arg(long)]
        models: PathBuf,

        /// Input image path
        #[arg(long)]
        image: PathBuf,

        /// Box prompt: x1 y1 x2 y2
        #[arg(long, num_args = 4, value_names = &["X1", "Y1", "X2", "Y2"])]
        r#box: Option<Vec<f32>>,

        /// Point prompt: x,y (can repeat, e.g. --point 475,380)
        #[arg(long, value_parser = parse_point_pair)]
        point: Vec<(f32, f32)>,

        /// Label for each point: fg or bg (default: fg)
        #[arg(long, value_parser = parse_label)]
        label: Vec<PointLabel>,

        /// Output image path
        #[arg(long, default_value = "output.jpg")]
        output: PathBuf,

        /// Delegate .so for INT8 models (e.g. libneutron_delegate.so)
        #[arg(long)]
        delegate: Vec<PathBuf>,

        /// Enable built-in XNNPACK delegate for attention (multi-threaded FP16)
        #[arg(long)]
        xnnpack: bool,

        /// Number of TFLite threads
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Which mask to use (0-3, default: highest IoU)
        #[arg(long)]
        mask_index: Option<usize>,

        /// Save all 4 individual mask images
        #[arg(long)]
        save_all_masks: bool,

        /// Warmup runs before timing
        #[arg(long, default_value_t = 1)]
        warmup: usize,

        /// Number of timed runs to average
        #[arg(long, default_value_t = 1)]
        runs: usize,
    },

    /// Benchmark the pipeline with timing breakdown
    Bench {
        /// Path to model archive ZIP
        #[arg(long)]
        models: PathBuf,

        /// Input image path
        #[arg(long)]
        image: PathBuf,

        /// Box prompt: x1 y1 x2 y2
        #[arg(long, num_args = 4, value_names = &["X1", "Y1", "X2", "Y2"])]
        r#box: Option<Vec<f32>>,

        /// Point prompt: x,y (can repeat, e.g. --point 475,380)
        #[arg(long, value_parser = parse_point_pair)]
        point: Vec<(f32, f32)>,

        /// Label for each point: fg or bg (default: fg)
        #[arg(long, value_parser = parse_label)]
        label: Vec<PointLabel>,

        /// Delegate .so for INT8 models
        #[arg(long)]
        delegate: Vec<PathBuf>,

        /// Enable built-in XNNPACK delegate for attention (multi-threaded FP16)
        #[arg(long)]
        xnnpack: bool,

        /// Number of TFLite threads
        #[arg(long, default_value_t = 0)]
        threads: usize,

        /// Warmup iterations
        #[arg(long, default_value_t = 5)]
        warmup: usize,

        /// Benchmark iterations
        #[arg(long, default_value_t = 20)]
        runs: usize,
    },

    /// List contents of a model archive
    Info {
        /// Path to model archive ZIP
        #[arg(long)]
        models: PathBuf,
    },
}

fn parse_point_pair(s: &str) -> Result<(f32, f32), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err(format!("Expected x,y but got '{}'", s));
    }
    let x: f32 = parts[0].trim().parse().map_err(|e| format!("Bad x: {}", e))?;
    let y: f32 = parts[1].trim().parse().map_err(|e| format!("Bad y: {}", e))?;
    Ok((x, y))
}

fn parse_label(s: &str) -> Result<PointLabel, String> {
    match s.to_lowercase().as_str() {
        "fg" | "foreground" | "1" => Ok(PointLabel::Foreground),
        "bg" | "background" | "0" => Ok(PointLabel::Background),
        _ => Err(format!("Invalid label '{}': use fg or bg", s)),
    }
}

fn build_prompt(
    box_coords: &Option<Vec<f32>>,
    points: &[(f32, f32)],
    labels: &[PointLabel],
) -> Result<Prompt, Box<dyn std::error::Error>> {
    if let Some(coords) = box_coords {
        if coords.len() != 4 {
            return Err("--box requires exactly 4 values: x1 y1 x2 y2".into());
        }
        return Ok(Prompt::Box(BBox {
            x1: coords[0],
            y1: coords[1],
            x2: coords[2],
            y2: coords[3],
        }));
    }

    if !points.is_empty() {
        let pts: Vec<Point> = points
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| Point {
                x,
                y,
                label: labels.get(i).copied().unwrap_or(PointLabel::Foreground),
            })
            .collect();
        return Ok(Prompt::Points(pts));
    }

    Err("A prompt is required: --box X1 Y1 X2 Y2 or --point X,Y".into())
}

fn build_config(
    models: &PathBuf,
    delegates: &[PathBuf],
    xnnpack: bool,
    threads: usize,
) -> SessionConfig {
    SessionConfig {
        model_archive: models.clone(),
        delegates: delegates.to_vec(),
        use_xnnpack: xnnpack,
        threads: if threads == 0 {
            num_cpus::get()
        } else {
            threads
        },
    }
}

fn print_timings(timings: &nanosam::Timings) {
    println!();
    println!("{:─<44}", "");
    println!("  {:<28} {:>8}", "Stage", "ms");
    println!("{:─<44}", "");
    println!("  {:<28} {:>8.1}", "Preprocess (resize+pad)", timings.preprocess_ms);
    println!("  {:<28} {:>8.1}", "Encoder (TFLite INT8)", timings.encoder_ms);
    println!("  {:<28} {:>8.1}", "Prompt encoder (Rust)", timings.prompt_encoder_ms);
    println!("  {:<28} {:>8.1}", "Attention (TFLite)", timings.attention_ms);
    println!("  {:<28} {:>8.1}", "Heads A+B (TFLite)", timings.heads_ms);
    println!("  {:<28} {:>8.1}", "Tokens (TFLite)", timings.tokens_ms);
    println!("  {:<28} {:>8.1}", "Mask assembly (matmul)", timings.mask_assembly_ms);
    println!("  {:<28} {:>8.1}", "Postprocess (upscale)", timings.postprocess_ms);
    println!("{:─<44}", "");
    println!("  {:<28} {:>8.1}", "Decoder total", timings.decoder_ms());
    println!("  {:<28} {:>8.1}", "Pipeline total", timings.total_ms());
    println!("{:─<44}", "");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Info { models } => {
            let archive = nanosam::archive::ModelArchive::load(&models)?;
            println!("Model archive: {}", models.display());
            println!("{:─<44}", "");
            for (name, size) in archive.list() {
                let size_str = if size > 1_000_000 {
                    format!("{:.1} MB", size as f64 / 1_048_576.0)
                } else {
                    format!("{:.1} KB", size as f64 / 1024.0)
                };
                println!("  {:<36} {:>8}", name, size_str);
            }
        }

        Command::Segment {
            models, image, r#box, point, label, output, delegate,
            xnnpack, threads, mask_index, save_all_masks, warmup, runs,
        } => {
            let prompt = build_prompt(&r#box, &point, &label)?;
            let config = build_config(&models, &delegate, xnnpack, threads);

            println!("Loading models from {}...", models.display());
            let t_load = Instant::now();
            let mut session = nanosam::Session::new(config)?;
            println!("  Loaded in {:.0} ms", t_load.elapsed().as_secs_f64() * 1000.0);

            println!("Loading image: {}", image.display());

            // Warmup (uses HAL preprocessing + low-res decode, no rendering)
            if warmup > 0 {
                println!("Warmup ({warmup} runs)...");
                for _ in 0..warmup {
                    session.segment_path(&image, &prompt)?;
                }
            }

            // Timed runs — use decode_low_res to skip CPU upscaling
            println!("Inference ({runs} runs)...");
            let mut low_res_masks = Vec::new();
            let mut best_idx = 0usize;
            let mut accumulated = nanosam::Timings::default();

            for _ in 0..runs {
                let t_pre = std::time::Instant::now();
                let (i8_data, original_hw) = if let Some(ref mut proc) = session.hal_processor {
                    nanosam::hal::preprocess_image_hal(proc, &image, &session.quant_coeffs)?
                } else {
                    // Fallback: use image crate path → float → quantize
                    let img = image::open(&image)?;
                    let (emb, pre_ms, enc_ms) = session.encoder.encode(&img)?;
                    accumulated.preprocess_ms += pre_ms;
                    accumulated.encoder_ms += enc_ms;
                    let r = session.decode(&emb, &prompt)?;
                    accumulated.prompt_encoder_ms += r.timings.prompt_encoder_ms;
                    accumulated.attention_ms += r.timings.attention_ms;
                    accumulated.heads_ms += r.timings.heads_ms;
                    accumulated.tokens_ms += r.timings.tokens_ms;
                    accumulated.mask_assembly_ms += r.timings.mask_assembly_ms;
                    accumulated.postprocess_ms += r.timings.postprocess_ms;
                    continue;
                };
                let preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

                let (emb, encoder_ms) = session.encoder.encode_i8(&i8_data, original_hw)?;
                let (masks, best, timings) = session.decode_low_res(&emb, &prompt)?;

                accumulated.preprocess_ms += preprocess_ms;
                accumulated.encoder_ms += encoder_ms;
                accumulated.prompt_encoder_ms += timings.prompt_encoder_ms;
                accumulated.attention_ms += timings.attention_ms;
                accumulated.heads_ms += timings.heads_ms;
                accumulated.tokens_ms += timings.tokens_ms;
                accumulated.mask_assembly_ms += timings.mask_assembly_ms;

                low_res_masks = masks;
                best_idx = best;
            }

            // Average timings
            let n = runs as f64;
            accumulated.preprocess_ms /= n;
            accumulated.encoder_ms /= n;
            accumulated.prompt_encoder_ms /= n;
            accumulated.attention_ms /= n;
            accumulated.heads_ms /= n;
            accumulated.tokens_ms /= n;
            accumulated.mask_assembly_ms /= n;

            // Render output using HAL (GPU mask overlay, no CPU upscale)
            let t_render = std::time::Instant::now();
            let idx = mask_index.unwrap_or(best_idx);

            {
                let proc = session.hal_processor.as_mut().expect("HAL required");
                let (mut dst, w, h) = nanosam::hal::load_image_tensor(proc, &image)?;
                println!("  Size: {}x{}", w, h);

                nanosam::hal::render_mask_hal(proc, &mut dst, &low_res_masks[idx], 0.4)?;
                nanosam::hal::save_tensor_jpeg(&dst, &output)?;
                println!("\n  Output: {}", output.display());

                if save_all_masks {
                    let stem = output.file_stem().unwrap().to_string_lossy();
                    let ext = output.extension().unwrap_or_default().to_string_lossy();
                    let parent = output.parent().unwrap_or(std::path::Path::new("."));

                    for (i, mask) in low_res_masks.iter().enumerate() {
                        let path = parent.join(format!("{}_mask_{}.{}", stem, i, ext));
                        let (mut img_dst, _, _) = nanosam::hal::load_image_tensor(proc, &image)?;
                        nanosam::hal::render_mask_hal(proc, &mut img_dst, mask, 0.4)?;
                        nanosam::hal::save_tensor_jpeg(&img_dst, &path)?;
                        let marker = if i == idx { " <-- selected" } else { "" };
                        println!("  mask_{} (IoU={:.3}): {}{}", i, mask.iou_score,
                                 path.display(), marker);
                    }
                }
            }
            accumulated.postprocess_ms = t_render.elapsed().as_secs_f64() * 1000.0 / n;

            print_timings(&accumulated);

            println!(
                "\n  IoU: [{}]",
                low_res_masks.iter().map(|m| format!("{:.3}", m.iou_score)).collect::<Vec<_>>().join(", ")
            );
            println!("  Best mask: index {} (IoU={:.3})", idx, low_res_masks[idx].iou_score);
        }

        Command::Bench {
            models, image, r#box, point, label, delegate,
            xnnpack, threads, warmup, runs,
        } => {
            let prompt = build_prompt(&r#box, &point, &label)?;
            let config = build_config(&models, &delegate, xnnpack, threads);

            println!("Loading models from {}...", models.display());
            let mut session = nanosam::Session::new(config)?;

            println!("Loading image: {}", image.display());

            // Take the HAL processor out of session to avoid borrow conflicts
            let mut proc = session.hal_processor.take().expect("HAL required for bench");
            let quant = session.quant_coeffs.clone();

            let (base_img, img_w, img_h) = nanosam::hal::load_image_tensor(&mut proc, &image)?;
            let mut render_dst = proc.create_image(
                img_w as usize, img_h as usize,
                edgefirst_tensor::PixelFormat::Rgba, edgefirst_tensor::DType::U8, None,
            )?;
            println!("  Size: {}x{}", img_w, img_h);

            println!("Warmup: {} iterations", warmup);
            for _ in 0..warmup {
                let (i8_data, hw) =
                    nanosam::hal::preprocess_image_hal(&mut proc, &image, &quant)?;
                let (emb, _) = session.encoder.encode_i8(&i8_data, hw)?;
                let (masks, best, _) = session.decode_low_res(&emb, &prompt)?;
                proc.convert(
                    &base_img, &mut render_dst,
                    edgefirst_image::Rotation::None, edgefirst_image::Flip::None,
                    edgefirst_image::Crop::new(),
                )?;
                nanosam::hal::render_mask_hal(&mut proc, &mut render_dst, &masks[best], 0.4)?;
            }

            println!("Benchmark: {} iterations", runs);
            let mut accumulated = nanosam::Timings::default();
            let mut last_masks = Vec::new();
            let mut last_best = 0usize;

            for _ in 0..runs {
                let t_pre = std::time::Instant::now();
                let (i8_data, hw) =
                    nanosam::hal::preprocess_image_hal(&mut proc, &image, &quant)?;
                let preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

                let (emb, encoder_ms) = session.encoder.encode_i8(&i8_data, hw)?;
                let (masks, best, timings) = session.decode_low_res(&emb, &prompt)?;

                // Reset render target to base image, then overlay mask
                let t_post = std::time::Instant::now();
                proc.convert(
                    &base_img, &mut render_dst,
                    edgefirst_image::Rotation::None, edgefirst_image::Flip::None,
                    edgefirst_image::Crop::new(),
                )?;
                nanosam::hal::render_mask_hal(&mut proc, &mut render_dst, &masks[best], 0.4)?;
                let postprocess_ms = t_post.elapsed().as_secs_f64() * 1000.0;

                accumulated.preprocess_ms += preprocess_ms;
                accumulated.encoder_ms += encoder_ms;
                accumulated.prompt_encoder_ms += timings.prompt_encoder_ms;
                accumulated.attention_ms += timings.attention_ms;
                accumulated.heads_ms += timings.heads_ms;
                accumulated.tokens_ms += timings.tokens_ms;
                accumulated.mask_assembly_ms += timings.mask_assembly_ms;
                accumulated.postprocess_ms += postprocess_ms;

                last_masks = masks;
                last_best = best;
            }

            let n = runs as f64;
            accumulated.preprocess_ms /= n;
            accumulated.encoder_ms /= n;
            accumulated.prompt_encoder_ms /= n;
            accumulated.attention_ms /= n;
            accumulated.heads_ms /= n;
            accumulated.tokens_ms /= n;
            accumulated.mask_assembly_ms /= n;
            accumulated.postprocess_ms /= n;

            print_timings(&accumulated);
            if runs > 1 {
                println!("  (averaged over {} runs)", runs);
            }

            // Save last rendered overlay for inspection
            let stem = image.file_stem().unwrap_or_default().to_string_lossy();
            let out_path = image.with_file_name(format!("{stem}_bench_overlay.jpg"));
            nanosam::hal::save_tensor_jpeg(&render_dst, &out_path)?;

            println!(
                "\n  IoU: [{}]",
                last_masks.iter().map(|m| format!("{:.3}", m.iou_score)).collect::<Vec<_>>().join(", ")
            );
            println!("  Best mask: index {} (IoU={:.3})", last_best, last_masks[last_best].iou_score);
            println!("  Overlay: {}", out_path.display());
        }
    }

    Ok(())
}
