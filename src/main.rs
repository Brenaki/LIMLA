/*
* author: Victor Cerqueira
* start: 09-04-2025
* last-update: 09-04-2025
*/

use image::DynamicImage;
use std::env;
use std::fs;

/*
* in: path input iamge
* out: diretory with images compressions "compressed_images/{name_file}_{% compression}.jpg"
*/
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <input_image.jpg>", args[0]);
        std::process::exit(1);
    }
    
    // path for file
    let input_path = &args[1];

    // open image
    let img = image::open(input_path)?;
    
    // vetor with levels about compression
    let quality_levels = vec![1, 2, 5, 10, 20, 30, 40, 50];
    
    // create diretory
    fs::create_dir_all("compressed_images")?;
    
    println!("Original image: {}", input_path);
    let original_size = fs::metadata(input_path)?.len();
    println!("Original size: {} bytes\n", original_size);

    // get name file
    let name_file = input_path.get(24..).unwrap_or("unknown");
    
    // call compress_jpeg for all levels compression
    for quality in quality_levels {
        let output_path = format!("compressed_images/{}_{}%.jpg", name_file, quality);
        compress_jpeg(&img, &output_path, quality)?;
        
        // differences with input file and output file
        let compressed_size = fs::metadata(&output_path)?.len();
        let compression_ratio = (compressed_size as f64 / original_size as f64) * 100.0;
        
        println!("Quality {}%: {} bytes ({:.1}% of original)", 
                quality, compressed_size, compression_ratio);
    }
    
    Ok(())
}

fn compress_jpeg(img: &DynamicImage, output_path: &str, quality: u8) -> Result<(), Box<dyn std::error::Error>> {
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
    fn test_compress_jpeg() -> Result<(), Box<dyn std::error::Error>> {
        // create a simple test image (100x100 pixels, red color)
        let test_image = image::RgbImage::from_fn(100, 100, |_, _| {
            image::Rgb([255, 0, 0])
        });
        let dynamic_image = image::DynamicImage::ImageRgb8(test_image);
        
        // test output path
        let output_path = "test_compressed.jpg";
        
        // test compression with quality 50
        let result = compress_jpeg(&dynamic_image, output_path, 50);
        assert!(result.is_ok(), "Compression should succeed");
        
        // verify the output file exists
        assert!(Path::new(output_path).exists(), "Output file should be created");
        
        // verify the file has some content (not empty)
        let metadata = fs::metadata(output_path)?;
        assert!(metadata.len() > 0, "Compressed file should not be empty");
        
        // clean up test file
        fs::remove_file(output_path)?;
        
        Ok(())
    }

    #[test]
    fn test_compress_jpeg_different_qualities() -> Result<(), Box<dyn std::error::Error>> {
        // create a test image
        let test_image = image::RgbImage::from_fn(200, 200, |x, y| {
            image::Rgb([x as u8, y as u8, 128])
        });
        let dynamic_image = image::DynamicImage::ImageRgb8(test_image);
        
        let qualities = vec![10, 50, 90];
        let mut file_sizes = Vec::new();
        
        // test compressin with quality 10, 50, and 90
        for (i, quality) in qualities.iter().enumerate() {
            let output_path = format!("test_compressed_{}.jpg", i);
            
            let result = compress_jpeg(&dynamic_image, &output_path, *quality);
            assert!(result.is_ok(), "Compression with quality {} should succeed", quality);
            
            let metadata = fs::metadata(&output_path)?;
            file_sizes.push(metadata.len());
            
            // clean up
            fs::remove_file(output_path)?;
        }
        
        // verify that lower quality produces smaller files
        assert!(file_sizes[0] < file_sizes[1], "Quality 10 should produce smaller file than quality 50");
        assert!(file_sizes[1] < file_sizes[2], "Quality 50 should produce smaller file than quality 90");
        
        Ok(())
    }

    #[test]
    fn test_compress_jpeg_invalid_path() {
        // test with invalid output path (directory that doesn't exist)
        let test_image = image::RgbImage::from_fn(50, 50, |_, _| {
            image::Rgb([255, 255, 255])
        });
        let dynamic_image = image::DynamicImage::ImageRgb8(test_image);
        
        let result = compress_jpeg(&dynamic_image, "/nonexistent/path/test.jpg", 50);
        assert!(result.is_err(), "Compression with invalid path should fail");
    }
}