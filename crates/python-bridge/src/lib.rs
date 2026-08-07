#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("[RustyLM-python] {}", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_err {
    ($($arg:tt)*) => {
        eprintln!("[RustyLM-python][ERROR] {}", format!($($arg)*));
    };
}