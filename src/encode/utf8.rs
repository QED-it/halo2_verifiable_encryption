/// Data type transformation functions
pub fn convert_u8_array_to_u64_array(input: [u8; 32]) -> [u64; 4] {
    let mut output = [0u64; 4];
    for (i, chunk) in input.chunks_exact(8).enumerate() {
        let bytes: [u8; 8] = chunk.try_into().expect("slice with incorrect length");
        output[i] = u64::from_le_bytes(bytes); // Use from_le_bytes if the input is little-endian
    }
    output
}

pub fn convert_string_to_u8_array(str: &str) -> [u8; 32] {
    let message_bytes = str.as_bytes();

    if message_bytes.len() > 32 {
        panic!("Message is too long to fit in buffer, at most 32 bytes!");
    }

    // Initialize padded_m with 32 zeroed bytes
    let mut padded_m: [u8; 32] = [0; 32];

    // copy msg into padded_m
    padded_m[..message_bytes.len()].copy_from_slice(&message_bytes);
    padded_m
}

pub fn convert_u8_array_to_string(arr: [u8; 32]) -> String {
    let bytes = arr
        .iter()
        // Take bytes while the byte is not zero
        .take_while(|&&byte| byte != 0)
        // Clone the bytes to create an owned Vec<u8>
        .cloned()
        // Collect bytes into a Vec<u8>
        .collect();
    String::from_utf8(bytes).expect("Invalid UTF-8")
}

pub(crate) fn split_message_into_blocks(message: &str, block_size: usize) -> Vec<String> {
    message
        .chars() // Work with chars to respect UTF-8 character boundaries
        .collect::<Vec<char>>() // Collect chars into a Vec to chunk by char count
        .chunks(block_size) // Split into chunks of `block_size` chars
        .map(|chunk| chunk.iter().collect::<String>()) // Collect each chunk of chars into a String
        .collect()
}
