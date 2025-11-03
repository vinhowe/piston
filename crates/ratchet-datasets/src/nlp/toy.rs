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

/// A task where the model needs to sort numbers after a "sort:" token
pub struct SortTask {
    seq_len: usize,
    rng: StdRng,
}

impl SortTask {
    pub fn new(seq_len: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { seq_len, rng }
    }
}

impl ToyTask for SortTask {
    fn generate_example(&mut self) -> String {
        let mut nums = Vec::with_capacity(self.seq_len);
        for _ in 0..self.seq_len {
            nums.push(self.rng.gen_range(0..10));
        }

        // Create a sorted copy for the target
        let mut sorted_nums = nums.clone();
        sorted_nums.sort();

        let nums_str: Vec<String> = nums.iter().map(|&n| format!("{}", n)).collect();
        let sorted_str: Vec<String> = sorted_nums.iter().map(|&n| format!("{}", n)).collect();

        // format!("{}:{}", nums_str.join(","), sorted_str.join(","))
        format!("{}:{}", nums_str.concat(), sorted_str.concat())
    }
}

/// A task where the model needs to add two numbers
pub struct AddTask {
    max_num: usize,
    rng: StdRng,
}

impl AddTask {
    pub fn new(max_num: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { max_num, rng }
    }
}

impl ToyTask for AddTask {
    fn generate_example(&mut self) -> String {
        let a = self.rng.gen_range(0..self.max_num);
        let b = self.rng.gen_range(0..self.max_num);
        let sum = a + b;

        // We want fixed-width examples
        let width = (self.max_num as f64).log10().floor() as usize + 1;

        format!("{:0width$}+{:0width$}={:0width$}", a, b, sum, width = width)
    }
}

/// A task where the model needs to count occurrences of a character in a string
pub struct CountTask {
    len: usize,
    max_char: char,
    rng: StdRng,
}

impl CountTask {
    pub fn new(len: usize, max_char: char, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { len, max_char, rng }
    }
}

impl ToyTask for CountTask {
    fn generate_example(&mut self) -> String {
        // Generate a random string length between 2 and max_len
        // Calculate range of valid characters (from 'a' to max_char inclusive)
        let char_range = self.max_char as u8 - b'a' + 1;

        // Generate random letters up to max_char
        let chars: Vec<char> = (0..self.len)
            .map(|_| (b'a' + self.rng.gen_range(0..char_range)) as char)
            .collect();

        // Pick a random character from the generated string to count
        let target_char = chars[self.rng.gen_range(0..self.len)];

        // Count occurrences
        let count = chars.iter().filter(|&&c| c == target_char).count();

        format!(
            "{}:{}={}",
            chars.iter().collect::<String>(),
            target_char,
            count
        )
    }
}

/// A task where the model needs to identify when to slap in a card game
/// Rules:
/// - Slap on Jack (J)
/// - Slap on doubles (same card twice in a row)
/// - Slap on sandwiches (same card with one card between)
pub struct SlapjackTask {
    seq_len: usize,
    rng: StdRng,
}

impl SlapjackTask {
    pub fn new(seq_len: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { seq_len, rng }
    }

    fn generate_card(&mut self) -> char {
        const CARDS: &[char] = &['A', '2', '3', '4', '5', '6', '7', '8', '9', 'J', 'Q', 'K'];
        CARDS[self.rng.gen_range(0..CARDS.len())]
    }
}

impl ToyTask for SlapjackTask {
    fn generate_example(&mut self) -> String {
        let mut sequence = Vec::new();
        let mut n_cards = 0;
        let mut n_slaps = 0;

        while sequence.len() - n_slaps < ((self.seq_len / 2) - 1) {
            let card = self.generate_card();
            sequence.push(card);

            if !sequence.is_empty() {
                let i = sequence.len() - 1;

                // Check slap conditions
                let card_is_jack = card == 'J';
                let card_is_double = n_cards > 0 && card == sequence[i - 1];
                let card_is_sandwich = n_cards > 1 && card == sequence[i - 2];

                let should_slap = card_is_jack || card_is_double || card_is_sandwich;

                if should_slap && sequence.len() < self.seq_len {
                    sequence.push('*');
                    n_cards = 0;
                    n_slaps += 1;
                    continue;
                }
            }

            n_cards += 1;
        }

        // Format as input=output where input is sequence without slaps
        // and output is sequence with slaps
        let input: String = sequence.iter().filter(|&&c| c != '*').collect();
        let output: String = sequence.iter().collect();

        let mut result = format!("{}={}", input, output);
        result.truncate(self.seq_len);
        result
    }
}

/// A task where the model needs to perform modular addition modulo a prime number
pub struct ModAddTask {
    prime: usize,
    rng: StdRng,
}

impl ModAddTask {
    pub fn new(prime: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { prime, rng }
    }
}

impl ToyTask for ModAddTask {
    fn generate_example(&mut self) -> String {
        let a = self.rng.gen_range(0..self.prime);
        let b = self.rng.gen_range(0..self.prime);
        let sum = (a as u16 + b as u16) % (self.prime as u16);

        // Get number of digits needed by finding floor(log10(prime)) + 1
        // We want fixed-width examples
        let width = (self.prime as f64).log10().floor() as usize + 1;

        format!("{:0width$}+{:0width$}={:0width$}", a, b, sum, width = width)
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

        // 5) Parse the two "final" numbers
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

    #[test]
    fn test_sort() {
        let mut sort_task = SortTask::new(100, Some(42));
        let example = sort_task.generate_example();

        // Split at the colon to get the command and data
        let (cmd, rest) = example
            .split_once(':')
            .expect("Expected a colon in sort format");
        assert_eq!(cmd, "sort", "Command should be 'sort'");

        // Split at equals to get input and output
        let (input_str, output_str) = rest.split_once('=').expect("Expected an equals sign");

        // Parse input and output numbers
        let input: Vec<u8> = input_str
            .split(',')
            .map(|s| s.parse::<u8>().expect("Failed parsing input number"))
            .collect();
        let output: Vec<u8> = output_str
            .split(',')
            .map(|s| s.parse::<u8>().expect("Failed parsing output number"))
            .collect();

        // Verify output is sorted
        let mut expected = input.clone();
        expected.sort();
        assert_eq!(output, expected, "Output should be sorted input");
    }

    #[test]
    fn test_add() {
        let mut add_task = AddTask::new(100, Some(42));
        let example = add_task.generate_example();

        // Split at the plus to get first number
        let (a_str, rest) = example.split_once('+').expect("Expected a plus sign");
        // Split at equals to get second number and result
        let (b_str, result_str) = rest.split_once('=').expect("Expected an equals sign");

        // Parse the numbers
        let a: u8 = a_str.parse().expect("Failed parsing first number");
        let b: u8 = b_str.parse().expect("Failed parsing second number");
        let result: u8 = result_str.parse().expect("Failed parsing result");

        // Verify the addition
        assert_eq!(a + b, result, "Addition result should be correct");
    }

    #[test]
    fn test_count() {
        let mut count_task = CountTask::new(10, 'f', Some(42));
        let example = count_task.generate_example();

        // Split at the colon to get the input string
        let (input_str, rest) = example.split_once(':').expect("Expected a colon");
        // Split at equals to get target char and count
        let (target_char_str, count_str) = rest.split_once('=').expect("Expected an equals sign");

        // Parse the target character and count
        let target_char = target_char_str
            .chars()
            .next()
            .expect("Expected a target character");
        let count: usize = count_str.parse().expect("Failed parsing count");

        // Verify the count
        let actual_count = input_str.chars().filter(|&c| c == target_char).count();
        assert_eq!(count, actual_count, "Count should match actual occurrences");
    }

    #[test]
    fn test_slapjack() {
        let mut slapjack_task = SlapjackTask::new(18, Some(42));
        let example = slapjack_task.generate_example();

        // Split at equals to get input and output sequences
        let (input, output) = example.split_once('=').expect("Expected an equals sign");

        // Verify that input sequence has no slaps
        assert!(
            !input.contains('*'),
            "Input sequence should not contain slaps"
        );

        // Verify that output sequence contains slaps
        let mut n_cards = 0;
        let output_chars: Vec<char> = output.chars().collect();

        for i in 0..output_chars.len() {
            let card = output_chars[i];
            if card == '*' {
                // Verify that the slap was valid
                let prev_card = output_chars[i - 1];
                let is_jack = prev_card == 'J';
                let is_double = n_cards > 0 && i > 0 && prev_card == output_chars[i - 1];
                let is_sandwich =
                    n_cards > 1 && i > 1 && prev_card != '*' && prev_card == output_chars[i - 2];

                assert!(
                    is_jack || is_double || is_sandwich,
                    "Invalid slap at position {}",
                    i
                );

                n_cards = 0;
            } else {
                n_cards += 1;
            }
        }
    }

    #[test]
    fn test_mod_add() {
        let mut mod_add_task = ModAddTask::new(7, Some(42));
        let example = mod_add_task.generate_example();

        // Split at the plus to get first number
        let (a_str, rest) = example.split_once('+').expect("Expected a plus sign");
        // Split at equals to get second number and result with modulus
        let (b_str, result_with_mod) = rest.split_once('=').expect("Expected an equals sign");
        // Split at space to get result and modulus
        let (result_str, modulus_part) = result_with_mod.split_once(' ').expect("Expected a space");
        // Extract modulus number
        let modulus: u8 = modulus_part[5..modulus_part.len() - 1]
            .parse()
            .expect("Failed parsing modulus");

        // Parse the numbers
        let a: u8 = a_str.parse().expect("Failed parsing first number");
        let b: u8 = b_str.parse().expect("Failed parsing second number");
        let result: u8 = result_str.parse().expect("Failed parsing result");

        // Verify the modular addition
        assert_eq!(
            (a as u16 + b as u16) % (modulus as u16),
            result as u16,
            "Modular addition result should be correct"
        );
        assert!(a < modulus, "First number should be less than modulus");
        assert!(b < modulus, "Second number should be less than modulus");
        assert!(result < modulus, "Result should be less than modulus");
    }
}
