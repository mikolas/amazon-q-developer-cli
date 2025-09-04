#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use crate::embedding::CandleTextEmbedder;
use crate::embedding::MockTextEmbedder; // Used for Fast type since BM25 doesn't need embeddings
use crate::embedding::OllamaTextEmbedder;
#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
use crate::embedding::ModelType;
use crate::embedding::{
    EmbeddingType,
    TextEmbedderTrait,
};
use crate::error::Result;
use crate::config::SemanticSearchConfig;

/// Creates a text embedder based on the specified embedding type
///
/// # Arguments
///
/// * `embedding_type` - Type of embedding engine to use
/// * `config` - Configuration containing Ollama and other settings
///
/// # Returns
///
/// A text embedder instance
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub fn create_embedder(embedding_type: EmbeddingType, config: &SemanticSearchConfig) -> Result<Box<dyn TextEmbedderTrait>> {
    let embedder: Box<dyn TextEmbedderTrait> = match embedding_type {
        EmbeddingType::Fast => Box::new(MockTextEmbedder::new(384)), // BM25 doesn't use embeddings
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        EmbeddingType::Best => Box::new(CandleTextEmbedder::with_model_type(ModelType::MiniLML6V2)?),
        EmbeddingType::Ollama => Box::new(OllamaTextEmbedder::new(
            &config.ollama_base_url,
            &config.ollama_model,
            config.ollama_timeout,
            config.ollama_batch_size,
        )?),
        #[cfg(test)]
        EmbeddingType::Mock => Box::new(MockTextEmbedder::new(384)),
    };

    Ok(embedder)
}

/// Creates a text embedder based on the specified embedding type
/// (Linux version)
///
/// # Arguments
///
/// * `embedding_type` - Type of embedding engine to use
/// * `config` - Configuration containing Ollama and other settings
///
/// # Returns
///
/// A text embedder instance
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn create_embedder(embedding_type: EmbeddingType, config: &SemanticSearchConfig) -> Result<Box<dyn TextEmbedderTrait>> {
    let embedder: Box<dyn TextEmbedderTrait> = match embedding_type {
        EmbeddingType::Fast => Box::new(MockTextEmbedder::new(384)), // BM25 doesn't use embeddings
        #[cfg(not(target_arch = "aarch64"))]
        EmbeddingType::Best => Box::new(CandleTextEmbedder::with_model_type(ModelType::MiniLML6V2)?),
        EmbeddingType::Ollama => Box::new(OllamaTextEmbedder::new(
            &config.ollama_base_url,
            &config.ollama_model,
            config.ollama_timeout,
            config.ollama_batch_size,
        )?),
        #[cfg(test)]
        EmbeddingType::Mock => Box::new(MockTextEmbedder::new(384)),
    };

    Ok(embedder)
}
