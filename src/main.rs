/*
* author: Victor Cerqueira
* start: 09-04-2025
* last-update: 17-12-2025
*/

use clap::{Parser, Subcommand, Args as ClapArgs};
use glob::glob;
use image::DynamicImage;
use indicatif::ProgressBar;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use rand::rng;
use std::path::{Path, PathBuf};
use std::process::Command;

/// LIMLA
#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    /// Path for dataset
    #[arg(short, long)]
    path: String,

    /// train % for split in folder (Example: 0.7)
    #[arg(short = 'r', long, default_value_t = 0.8)]
    train: f32,

    /// val % for split in folder (Example: 0.15)
    #[arg(short, long, default_value_t = 0.1)]
    val: f32,

    /// test % for split in folder (Example: 0.15)
    #[arg(short, long, default_value_t = 0.1)]
    test: f32,

    /// quality levels to compress (Example: "1,5,10")
    #[arg(short, long, default_value = "1,5,10")]
    quality: String,

    /// output directory (Example: "./compressed")
    #[arg(short, long, default_value = "./compressed")]
    output: String,

    /// call cnn to train
    #[command(subcommand)]
    cnn: Option<CnnCommand>,
}

#[derive(Subcommand, Debug)]
enum CnnCommand {
    /// Train CNN model
    Run(CnnRunArgs),
}

#[derive(ClapArgs, Debug)]
struct CnnRunArgs {
    /// model to train (Example: "MobileNetV2")
    #[arg(short, long, default_value = "MobileNetV2")]
    model: String,

    /// epochs to train (Example: 50)
   #[arg(short, long, default_value_t = 1)]
    epochs: u32,

    /// batch size to train (Example: 32)
    #[arg(short, long, default_value_t = 8)]
    batch_size: u32,

    /// patience to train (Example: 3)
    #[arg(short, long, default_value_t = 3)]
    patience: u32,

    /// output directory to train (Example: "./out")
    #[arg(short, long, default_value = "./out")]
    output_dir: String,
}

// parse quality levels from CLI
fn parse_quality_levels(quality_str: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    quality_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<u8>()
                .map_err(|e| format!("Invalid quality level '{}': {}", s, e).into())
        })
        .collect()
}

// config split (train, val, test)
struct SplitConfig {
    train: f32,
    val: f32,
    test: f32,
}

impl Default for SplitConfig {
    fn default() -> Self {
        SplitConfig {
            train: 0.8,
            val: 0.1,
            test: 0.1,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    // path for directory
    let input_path = &args.path;

    // parse split percentafes from args or use defaults
    let split_config = if args.train + args.val + args.test == 1.0 {
        SplitConfig { train: args.train, val: args.val, test: args.test }
    } else {
        SplitConfig::default()
    };

    println!(
        "Split configuration: train={:.0}%, val={:.0}%, test={:.0}%",
        split_config.train * 100.0,
        split_config.val * 100.0,
        split_config.test * 100.0
    );

    // parse quality levels from CLI
    let quality_levels = parse_quality_levels(&args.quality)?;
    println!("Quality levels: {:?}", quality_levels);

    // get all images per classes
    let images_by_class = collet_images_by_class(input_path)?;

    if images_by_class.is_empty() {
        eprintln!("No images found in {}", input_path);
        std::process::exit(1);
    }

    println!("Found {} classes", images_by_class.len());
    for (class, images) in &images_by_class {
        println!(" - {}: {} images", class, images.len());
    }

    let total_images = images_by_class.values().map(|v| v.len()).sum::<usize>();
    let total_operations = total_images * quality_levels.len();
    let bar = ProgressBar::new(total_operations as u64);

    // process per classe
    for (class_name, images) in images_by_class {
        split_and_compress_class(&class_name, images, &split_config, &quality_levels, &bar, &args.output)?;
    }

    bar.finish_with_message("Compression complete!");

    if let Some(CnnCommand::Run(cnn_args)) = args.cnn {
        train_cnn(&args.output, &quality_levels, &cnn_args)?;
    }

    Ok(())
}

// train cnn
fn train_cnn(output: &str, quality_levels: &[u8], cnn_args: &CnnRunArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Convert output path to be relative to cnn directory
    let data_dir = if output.starts_with('/') {
        output.to_string()
    } else {
        format!("../{}", output)
    };
    
    let command = format!(
        "cd cnn && source .venv/bin/activate && python main.py --model {} --data_dir {} --quality {} --epochs {} --batch_size {} --output_dir {} --patience {}",
        cnn_args.model,
        data_dir,
        quality_levels[0],
        cnn_args.epochs,
        cnn_args.batch_size,
        cnn_args.output_dir,
        cnn_args.patience
    );

    let mut child = Command::new("bash")
        .arg("-c")
        .arg(&command)
        .spawn()
        .expect("process failed to start");

    let status = child.wait().expect("failed to wait for process");

    if !status.success() {
        return Err(format!("CNN training failed with exit code: {:?}", status.code()).into());
    }

    println!("CNN trained successfully");

    Ok(())
}

// collect images by class
fn collet_images_by_class(
    input_path: &str,
) -> Result<HashMap<String, Vec<PathBuf>>, Box<dyn std::error::Error + Send + Sync>> {
    let mut images_by_class: HashMap<String, Vec<PathBuf>> = HashMap::new();

    // search all images JPEG (both .jpg and .jpeg)
    let patterns = vec![
        format!("{}/**/*.jpg", input_path),
        format!("{}/**/*.jpeg", input_path),
    ];

    let mut paths = Vec::new();
    for pattern in patterns {
        let found: Vec<_> = glob(&pattern)
            .expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .collect();
        paths.extend(found);
    }

    for path in paths {
        if let Some(parent) = path.parent() {
            if let Some(class_name) = parent.file_name() {
                let class_str = class_name.to_string_lossy().to_string();
                images_by_class
                    .entry(class_str)
                    .or_insert_with(Vec::new)
                    .push(path);
            }
        }
    }

    Ok(images_by_class)
}

// split and compress class
fn split_and_compress_class(
    class_name: &str,
    mut images: Vec<PathBuf>,
    config: &SplitConfig,
    quality_levels: &[u8],
    bar: &ProgressBar,
    output: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut rng = rng();
    images.as_mut_slice().shuffle(&mut rng);

    let total = images.len();
    let train_count = (total as f32 * config.train) as usize;
    let val_count = (total as f32 * config.val) as usize;

    let train_imgs = &images[0..train_count];
    let val_imgs = &images[train_count..train_count + val_count];
    let test_imgs = &images[train_count + val_count..];

    process_split(train_imgs, "train", class_name, quality_levels, bar, output)?;
    process_split(val_imgs, "val", class_name, quality_levels, bar, output)?;
    process_split(test_imgs, "test", class_name, quality_levels, bar, output)?;

    Ok(())
}

// process split
fn process_split(
    images: &[PathBuf],
    split: &str,
    class: &str,
    quality_levels: &[u8],
    bar: &ProgressBar,
    output: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    images.par_iter().try_for_each(|img_path| {
        compress_image_to_split(img_path, split, class, quality_levels, bar, output)?;
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })?;

    Ok(())
}

// compress image to split
fn compress_image_to_split(
    img_path: &Path,
    split: &str,
    class: &str,
    quality_levels: &[u8],
    bar: &ProgressBar,
    output: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let img = image::open(img_path)?;

    let file_name = img_path
        .file_stem()
        .ok_or("Invalid file name")?
        .to_string_lossy()
        .to_string();

    // Each thread processes a different quality level
    quality_levels.par_iter().try_for_each(
        |&quality| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let out_dir = format!("{}/q{}/{}/{}", output, quality, split, class);
            fs::create_dir_all(&out_dir)?;

            let output_path = format!("{}/{}.jpg", out_dir, file_name);
            compress_jpeg(&img, &output_path, quality)?;

            // Increment the progress bar for each quality level processed
            bar.inc(1);

            Ok(())
        },
    )?;

    Ok(())
}

// compress jpeg
fn compress_jpeg(
    img: &DynamicImage,
    output_path: &str,
    quality: u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::fs::File;
    use std::io::BufWriter;

    // path output
    let file = File::create(output_path)?;

    // buffer for writer bytes
    let mut writer = BufWriter::new(file);

    // encoder jpg for compression image
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, quality);

    // compression
    encoder.encode_image(img)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_compress_jpeg() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let test_image = image::RgbImage::from_fn(100, 100, |_, _| image::Rgb([255, 0, 0]));
        let dynamic_image = image::DynamicImage::ImageRgb8(test_image);
        let output_path = "test_compressed.jpg";

        let result = compress_jpeg(&dynamic_image, output_path, 50);
        assert!(result.is_ok(), "Compression should succeed");
        assert!(
            Path::new(output_path).exists(),
            "Output file should be created"
        );

        let metadata = fs::metadata(output_path)?;
        assert!(metadata.len() > 0, "Compressed file should not be empty");

        fs::remove_file(output_path)?;
        Ok(())
    }

    #[test]
    fn test_compress_jpeg_different_qualities()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let test_image =
            image::RgbImage::from_fn(200, 200, |x, y| image::Rgb([x as u8, y as u8, 128]));
        let dynamic_image = image::DynamicImage::ImageRgb8(test_image);

        let qualities = vec![10, 50, 90];
        let mut file_sizes = Vec::new();

        for (i, quality) in qualities.iter().enumerate() {
            let output_path = format!("test_compressed_{}.jpg", i);
            let result = compress_jpeg(&dynamic_image, &output_path, *quality);
            assert!(
                result.is_ok(),
                "Compression with quality {} should succeed",
                quality
            );

            let metadata = fs::metadata(&output_path)?;
            file_sizes.push(metadata.len());
            fs::remove_file(output_path)?;
        }

        assert!(
            file_sizes[0] < file_sizes[1],
            "Quality 10 should produce smaller file than quality 50"
        );
        assert!(
            file_sizes[1] < file_sizes[2],
            "Quality 50 should produce smaller file than quality 90"
        );

        Ok(())
    }

    #[test]
    fn test_compress_jpeg_invalid_path() {
        let test_image = image::RgbImage::from_fn(50, 50, |_, _| image::Rgb([255, 255, 255]));
        let dynamic_image = image::DynamicImage::ImageRgb8(test_image);

        let result = compress_jpeg(&dynamic_image, "/nonexistent/path/test.jpg", 50);
        assert!(result.is_err(), "Compression with invalid path should fail");
    }

    #[test]
    fn test_split_config_default() {
        let config = SplitConfig::default();
        assert_eq!(config.train, 0.8);
        assert_eq!(config.val, 0.1);
        assert_eq!(config.test, 0.1);
    }
}

