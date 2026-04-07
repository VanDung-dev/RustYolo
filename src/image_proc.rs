use ndarray::Array4;
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::resize::resize_fast_rgb;
use kornia_imgproc::interpolation::InterpolationMode;

/// Prepares the raw image buffer for YOLO inference using Kornia.
/// This includes creating a Kornia image, resizing, and converting to an ndarray tensor.
pub fn preprocess_image_kornia(
    raw_data: &[u8],
    orig_width: usize,
    orig_height: usize,
    target_width: usize,  
    target_height: usize,
) -> Result<Array4<f32>, String> {
    // 1. Create a Kornia Image from the raw slice.
    let image_size = ImageSize {
        width: orig_width,
        height: orig_height,
    };
    
    // Use from_size_slice instead of from_slice
    let image = Image::<u8, 3, CpuAllocator>::from_size_slice(image_size, raw_data, CpuAllocator)
        .map_err(|e| format!("Failed to create Kornia image: {:?}", e))?;

    // 2. Resize the image.
    let new_size = ImageSize {
        width: target_width,
        height: target_height,
    };
    
    // Kornia usually prefers pre-allocated output buffers
    let mut resized_image = Image::<u8, 3, CpuAllocator>::from_size_val(new_size, 0, CpuAllocator)
        .map_err(|e| format!("Failed to create output Kornia image: {:?}", e))?;
        
    resize_fast_rgb(
        &image,
        &mut resized_image,
        InterpolationMode::Bilinear,
    ).map_err(|e| format!("Failed to resize image with Kornia: {:?}", e))?;

    // 3. Normalize and transfer to ndarray.
    // YOLOv8 expects NCHW format: (1, 3, height, width).
    // Kornia Image is HWC [0, 255]. We need to transpose to CHW and normalize to [0, 1].
    
    // Convert to ndarray from raw slice (HWC)
    let resized_data = resized_image.as_slice();
    
    // Create an ndarray view of the HWC data
    let hwc_array = ndarray::ArrayView3::from_shape(
        (target_height, target_width, 3),
        resized_data
    ).map_err(|e| format!("Failed to create ndarray view: {:?}", e))?;

    // Perform transpose (HWC -> CHW) and normalization in one go using ndarray
    // This is much faster than manual loops and leverages SIMD
    let mut input_array = Array4::<f32>::zeros((1, 3, target_height, target_width));
    
    // We can use bitwise/vectorized operations by transposing the view
    let chw_array = hwc_array.permuted_axes([2, 0, 1]);
    
    // Assign and normalize
    input_array.slice_mut(ndarray::s![0, .., .., ..])
        .assign(&chw_array.mapv(|x| x as f32 / 255.0));

    Ok(input_array)
}
