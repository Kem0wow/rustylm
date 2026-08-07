pub mod device;
pub mod loader;

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("[RustyLM-core] {}", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_err {
    ($($arg:tt)*) => {
        eprintln!("[RustyLM-core][ERROR] {}", format!($($arg)*));
    };
}