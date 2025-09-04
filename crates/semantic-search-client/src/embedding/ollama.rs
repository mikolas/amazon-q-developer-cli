use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::runtime::Handle;
use tokio::sync::Semaphore;
use tracing::{debug, error};

use crate::error::{Result, SemanticSearchError};

#[derive(Serialize)]
struct OllamaEmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

/// Ollama-based text embedder that communicates with external Ollama server
pub struct OllamaTextEmbedder {
    client: Client,
    base_url: String,
    model: String,
    timeout: Duration,
    batch_size: usize,
    runtime_handle: Handle,
}

impl OllamaTextEmbedder {
    /// Create a new OllamaTextEmbedder
    pub fn new(base_url: &str, model: &str, timeout_ms: u64, batch_size: usize) -> Result<Self> {
        let timeout = Duration::from_millis(timeout_ms);
        
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| SemanticSearchError::EmbeddingError(format!("Failed to create HTTP client: {}", e)))?;

        // Get current runtime handle or create error if not in async context
        let runtime_handle = Handle::try_current()
            .map_err(|_| SemanticSearchError::EmbeddingError("No tokio runtime available".to_string()))?;

        debug!("Created Ollama embedder for model {} at {}", model, base_url);

        Ok(Self {
            client,
            base_url: base_url.to_string(),
            model: model.to_string(),
            timeout,
            batch_size,
            runtime_handle,
        })
    }

    /// Generate embedding for a single text (async version)
    async fn embed_async(&self, text: &str) -> Result<Vec<f32>> {
        let request = OllamaEmbedRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let url = format!("{}/api/embeddings", self.base_url);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send request to Ollama: {}", e);
                SemanticSearchError::EmbeddingError(format!("Ollama request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            error!("Ollama API error {}: {}", status, error_text);
            return Err(SemanticSearchError::EmbeddingError(format!(
                "Ollama API error {}: {}", status, error_text
            )));
        }

        let embed_response: OllamaEmbedResponse = response
            .json()
            .await
            .map_err(|e| {
                error!("Failed to parse Ollama response: {}", e);
                SemanticSearchError::EmbeddingError(format!("Invalid Ollama response: {}", e))
            })?;

        Ok(embed_response.embedding)
    }

    /// Generate embeddings for multiple texts (async version)
    async fn embed_batch_async(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Use semaphore to limit concurrent requests
        let semaphore = Arc::new(Semaphore::new(self.batch_size));
        let mut handles = Vec::new();

        // Clone necessary fields for sharing across tasks
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        let model = self.model.clone();

        for text in texts {
            let text = text.clone();
            let semaphore = semaphore.clone();
            let client = client.clone();
            let base_url = base_url.clone();
            let model = model.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.map_err(|e| {
                    SemanticSearchError::EmbeddingError(format!("Semaphore error: {}", e))
                })?;
                
                // Inline embed_async logic
                let request = OllamaEmbedRequest {
                    model,
                    prompt: text,
                };

                let url = format!("{}/api/embeddings", base_url);
                
                let response = client
                    .post(&url)
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| {
                        error!("Failed to send request to Ollama: {}", e);
                        SemanticSearchError::EmbeddingError(format!("Ollama request failed: {}", e))
                    })?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    error!("Ollama API error {}: {}", status, error_text);
                    return Err(SemanticSearchError::EmbeddingError(format!(
                        "Ollama API error {}: {}", status, error_text
                    )));
                }

                let embed_response: OllamaEmbedResponse = response
                    .json()
                    .await
                    .map_err(|e| {
                        error!("Failed to parse Ollama response: {}", e);
                        SemanticSearchError::EmbeddingError(format!("Invalid Ollama response: {}", e))
                    })?;

                Ok(embed_response.embedding)
            });
            
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut results = Vec::with_capacity(texts.len());
        for handle in handles {
            let result = handle.await.map_err(|e| {
                error!("Task join error: {}", e);
                SemanticSearchError::EmbeddingError(format!("Task failed: {}", e))
            })??;
            results.push(result);
        }

        Ok(results)
    }
}

impl crate::embedding::TextEmbedderTrait for OllamaTextEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Use block_in_place to run async code in sync context
        tokio::task::block_in_place(|| {
            self.runtime_handle.block_on(self.embed_async(text))
        })
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        tokio::task::block_in_place(|| {
            self.runtime_handle.block_on(self.embed_batch_async(texts))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    #[tokio::test]
    async fn test_ollama_embedder_creation() {
        let embedder = OllamaTextEmbedder::new(
            "http://localhost:11434",
            "nomic-embed-text",
            30000,
            32
        );
        
        assert!(embedder.is_ok());
    }

    #[test]
    fn test_ollama_embedder_without_runtime() {
        // This should fail when not in async context
        let result = OllamaTextEmbedder::new(
            "http://localhost:11434",
            "nomic-embed-text",
            30000,
            32
        );
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embed_request_structure() {
        let request = OllamaEmbedRequest {
            model: "test-model".to_string(),
            prompt: "test text".to_string(),
        };
        
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("test-model"));
        assert!(json.contains("test text"));
    }
}
