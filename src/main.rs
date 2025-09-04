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
