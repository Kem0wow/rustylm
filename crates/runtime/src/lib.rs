#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("[RustyLM-runtime] {}", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_err {
    ($($arg:tt)*) => {
        eprintln!("[RustyLM-runtime][ERROR] {}", format!($($arg)*));
    };
}