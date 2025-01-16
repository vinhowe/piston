use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use ratchet::shape;
use ratchet::Device;
use ratchet::Tensor;

pub trait ToyTask {
    /// Generate a single example of the toy task.
    fn generate_example(&mut self) -> String;
}

pub struct ToyTaskIter<T: ToyTask> {
    task: T,
    device: Device,
}

impl<T: ToyTask> ToyTaskIter<T> {
    pub fn new(task: T, device: Device) -> Self {
        Self { task, device }
    }
}

impl<T: ToyTask> Iterator for ToyTaskIter<T> {
    type Item = anyhow::Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        // Generate the next raw string from the toy task
        let example_str = self.task.generate_example();
        let bytes: Vec<_> = example_str.to_string().bytes().map(|b| b as i32).collect();

        // For an autoregressive input/target, we need at least 2 tokens
        if bytes.len() < 2 {
            return None;
        }

        // If we have N tokens, input is [0..N-1], target is [1..N]
        let seq_len = bytes.len() - 1;
        let inputs = Tensor::from_data(&bytes[..seq_len], shape![seq_len], self.device.clone());
        let targets = Tensor::from_data(&bytes[1..], shape![seq_len], self.device.clone());

        Some(Ok((inputs, targets)))
    }
}

/// "2-sum sequences" with optional seeding.
pub struct TwoSumTask {
    max_num: u8,
    seq_len: usize,
    rng: StdRng,
}

impl TwoSumTask {
    pub fn new(max_num: u8, seq_len: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            max_num,
            seq_len,
            rng,
        }
    }
}

impl ToyTask for TwoSumTask {
    fn generate_example(&mut self) -> String {
        let mut nums = Vec::with_capacity(self.seq_len);
        for _ in 0..self.seq_len {
            nums.push(self.rng.gen_range(0..self.max_num));
        }

        let i = self.rng.gen_range(0..self.seq_len);
        let j = self.rng.gen_range(0..self.seq_len);
        let target = nums[i] as u16 + nums[j] as u16;

        let nums_str: Vec<String> = nums.iter().map(|&n| format!("{:02}", n)).collect();
        format!(
            "{}:{:03}={:02},{:02}",
            nums_str.join(","),
            target,
            nums[i],
            nums[j]
        )
    }
}

/// For debugging more than anything, a sequence of zeros.
pub struct ZerosTask {
    seq_len: usize,
}

impl ZerosTask {
    pub fn new(seq_len: usize) -> Self {
        Self { seq_len }
    }
}

impl ToyTask for ZerosTask {
    fn generate_example(&mut self) -> String {
        "\x00".repeat(self.seq_len)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn test_twosum() {
        // Create the seeded TwoSumTask iterator
        let mut two_sum_seeded = TwoSumTask::new(100, 5, Some(42));

        // Get exactly one example
        let ex = two_sum_seeded.generate_example();

        // e.g. "12,35,07,99,03:134=35,99"

        // 1) Split at the colon to separate "12,35,07,99,03" from "134=35,99"
        let (nums_part, rest) = ex
            .split_once(':')
            .expect("Expected a colon ':' in the 2-sum format.");

        // 2) Split `rest` at '=' to separate "134" from "35,99"
        let (target_str, final_vals) = rest
            .split_once('=')
            .expect("Expected an '=' in the 2-sum format.");

        // 3) Parse the list of initial numbers
        let nums: Vec<u16> = nums_part
            .split(',')
            .map(|s| {
                s.parse::<u16>()
                    .expect("Failed parsing initial number as u16")
            })
            .collect();

        // 4) Parse the target
        let target: u16 = target_str
            .parse()
            .expect("Failed parsing the target sum as u16.");

        // 5) Parse the two “final” numbers
        let final_nums: Vec<u16> = final_vals
            .split(',')
            .map(|s| {
                s.parse::<u16>()
                    .expect("Failed parsing final numbers as u16")
            })
            .collect();

        // We expect exactly two final numbers
        assert_eq!(final_nums.len(), 2, "Expected exactly 2 final numbers.");
        let (val1, val2) = (final_nums[0], final_nums[1]);

        // Check sum
        assert_eq!(
            val1 + val2,
            target,
            "The two final numbers do not sum up to the target."
        );

        // Check that val1 and val2 are in the original list of numbers
        assert!(
            nums.contains(&val1) && nums.contains(&val2),
            "Expected both final numbers to appear in the initial list."
        );
    }

    #[test]
    fn test_zeros() {
        let mut zeros = ZerosTask::new(5);
        let example = zeros.generate_example();
        assert_eq!(example, "\x00\x00\x00\x00\x00");
        assert_eq!(example.len(), 5);
    }
}
