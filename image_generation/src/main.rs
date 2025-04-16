use tch::{Device, Kind, Tensor}; // Removed unused `std::ops::Div`
use image::{RgbImage, Rgb};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load image
    let img_path = "input.jpg";
    let img = image::open(img_path)?.to_rgb8();
    let (width, height) = img.dimensions();

    // Convert image to tensor (NCHW, normalized)
    let img_data = img.as_raw(); // &[u8]
    let img_tensor = Tensor::f_from_slice(img_data)?
        .view([height as i64, width as i64, 3])
        .permute([2, 0, 1]) // HWC -> CHW
        .to_kind(Kind::Float);

    // Explicitly create a tensor from 255.0_f32
    let divisor = Tensor::from(255.0_f32);
    let img_tensor = img_tensor / divisor;  // Divide by the tensor

    let img_tensor = img_tensor.unsqueeze(0); // Add batch dimension

    // Load TorchScript model
    let model = tch::CModule::load("yolov11m-face.torchscript")?;
    let output = model.forward_ts(&[img_tensor.to(Device::Cpu)])?; // Already a Tensor

    // [N, 6]: x1, y1, x2, y2, conf, class
    let detections = if output.size()[0] == 1 {
        output.squeeze_dim(0)  // Remove batch dimension if it exists
    } else {
        output
    };

    let mut img_buffer: RgbImage = img.clone();

    for i in 0..detections.size()[0] {
        let detection = detections.get(i);
        let x1 = detection.double_value(&[0]) as u32;
        let y1 = detection.double_value(&[1]) as u32;
        let x2 = detection.double_value(&[2]) as u32;
        let y2 = detection.double_value(&[3]) as u32;
        let _conf = detection.double_value(&[4]);
        let class_id = detection.double_value(&[5]) as u32;

        draw_rect(&mut img_buffer, x1, y1, x2, y2, Rgb([255, 0, 0]));
        println!("Detected class {} at [{}, {}, {}, {}]", class_id, x1, y1, x2, y2);
    }

    img_buffer.save("output.jpg")?;
    println!("Saved output to output.jpg");

    Ok(())
}

fn draw_rect(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>) {
    for x in x1..=x2 {
        if y1 < img.height() {
            img.put_pixel(x.min(img.width() - 1), y1, color);
        }
        if y2 < img.height() {
            img.put_pixel(x.min(img.width() - 1), y2, color);
        }
    }
    for y in y1..=y2 {
        if x1 < img.width() {
            img.put_pixel(x1, y.min(img.height() - 1), color);
        }
        if x2 < img.width() {
            img.put_pixel(x2, y.min(img.height() - 1), color);
        }
    }
}
