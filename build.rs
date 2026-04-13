//! Hỗ trợ rpath cho các hệ điều hành Unix-like (macOS, Linux)
//! trong quá trình phát triển và production.

fn main() {
    // Áp dụng cho các hệ điều hành Unix-like (macOS, Linux)
    #[cfg(unix)]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        
        // 1. Thêm rpath tuyệt đối cho môi trường phát triển (maturin develop)
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/target/release", manifest_dir);
        
        // 2. Thêm rpath tương đối cho môi trường production
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        } else if cfg!(target_os = "linux") {
            // Lưu ý: $ORIGIN cần viết đúng định dạng để linker hiểu
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
        
        println!("cargo:warning=🔗 Unix: Đã cấu hình rpath cho development tại target/release");
    }

    // Luôn yêu cầu rebuild nếu build.rs thay đổi
    println!("cargo:rerun-if-changed=build.rs");
}
