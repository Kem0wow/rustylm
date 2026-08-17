use std::error::Error;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let inner = HfTokenizer::from_file(path)?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn Error + Send + Sync>> {
        let encoding = self.inner.encode(text, add_special_tokens)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn Error + Send + Sync>> {
        let decoded = self.inner.decode(ids, skip_special_tokens)?;
        Ok(decoded)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_missing_file() {
        assert!(Tokenizer::load("non_existent_tokenizer.json").is_err());
    }
}