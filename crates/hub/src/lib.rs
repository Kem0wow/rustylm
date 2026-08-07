#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("[RustyLM-hub] {}", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_err {
    ($($arg:tt)*) => {
        eprintln!("[RustyLM-hub][ERROR] {}", format!($($arg)*));
    };
}