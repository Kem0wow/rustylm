use std::cell::Cell;
use std::time::{SystemTime, UNIX_EPOCH};

const TOP_K: usize = 64;

pub fn repeat_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &t in recent {
        if let Some(l) = logits.get_mut(t as usize) {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

pub fn sample(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }

    let mut top: Vec<(f32, u32)> = Vec::with_capacity(TOP_K + 1);
    let mut floor = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if top.len() == TOP_K && l <= floor {
            continue;
        }
        let at = top.partition_point(|&(p, _)| p > l);
        top.insert(at, (l, i as u32));
        top.truncate(TOP_K);
        floor = top.last().unwrap().0;
    }

    let max = top[0].0;
    let mut sum = 0f32;
    for (p, _) in top.iter_mut() {
        *p = ((*p - max) / temperature).exp();
        sum += *p;
    }

    let mut mass = 0f32;
    let mut keep = 0;
    for (p, _) in top.iter() {
        mass += *p / sum;
        keep += 1;
        if mass >= top_p {
            break;
        }
    }

    let target = rng() * top[..keep].iter().map(|(p, _)| *p).sum::<f32>();
    let mut acc = 0f32;
    for &(p, id) in &top[..keep] {
        acc += p;
        if acc >= target {
            return id;
        }
    }
    top[0].1
}

pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn rng() -> f32 {
    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(42);
            (t as u64).wrapping_add(1).wrapping_mul(6364136223846793005)
        });
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_picks_the_largest_logit() {
        let logits = [0.1, 9.0, -3.0, 2.0];
        assert_eq!(sample(&logits, 0.0, 1.0), 1);
        assert_eq!(sample(&logits, 0.01, 1.0), 1);
    }
}
