pub mod ops;

pub fn name() -> String {
    #[cfg(target_arch = "x86_64")]
    if let Some(cpuid_name) = get_cpuid_name() {
        return cpuid_name;
    }

    #[cfg(target_os = "linux")]
    if let Some(proc_name) = get_proc_cpuinfo_name() {
        return proc_name;
    }

    "Generic CPU".to_string()
}

#[cfg(target_arch = "x86_64")]
fn get_cpuid_name() -> Option<String> {
    use std::arch::x86_64::__cpuid;
    unsafe {
        if __cpuid(0x80000000).eax < 0x80000004 {
            return None;
        }
        let mut bytes = [0u8; 48];
        for (i, &leaf) in [0x80000002u32, 0x80000003, 0x80000004].iter().enumerate() {
            let res = __cpuid(leaf);
            let leaf_bytes: [u8; 16] = std::mem::transmute([res.eax, res.ebx, res.ecx, res.edx]);
            bytes[i * 16..(i + 1) * 16].copy_from_slice(&leaf_bytes);
        }
        let s = std::str::from_utf8(&bytes).ok()?.trim_matches('\0').trim();
        if !s.is_empty() {
            return Some(s.to_string());
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn get_proc_cpuinfo_name() -> Option<String> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in content.lines() {
        if line.starts_with("model name") || line.starts_with("Processor") {
            if let Some((_, val)) = line.split_once(':') {
                let name = val.trim();
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}