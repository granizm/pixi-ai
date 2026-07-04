//! The tool interface.
//!
//! A [`Tool`] is how the agent answers "did we have X? what about Y?" — each
//! tool exposes a capability (look something up, check state, query data) that
//! the model can invoke mid-conversation. The agent presents every registered
//! tool's [`ToolDefinition`] to the provider, and when the model emits a
//! `tool_use`, the agent dispatches to the matching tool's [`Tool::call`].

use async_trait::async_trait;

use llm::ToolDefinition;

/// Error returned by a tool invocation.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ToolError(pub String);

/// A capability the model can invoke during the agentic loop.
///
/// `Send + Sync` so tools can be held behind `Arc<dyn Tool>` and shared across
/// the agent and (eventually) the FFI boundary.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The schema advertised to the model (name, description, JSON Schema input).
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the model-provided input, returning a textual
    /// result that is fed back as a `tool_result`.
    async fn call(&self, input: serde_json::Value) -> Result<String, ToolError>;
}

/// A registry of tools the agent can dispatch to, keyed by tool name.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<std::sync::Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool.
    pub fn register(&mut self, tool: std::sync::Arc<dyn Tool>) {
        self.tools.push(tool);
    }

    /// All registered tool definitions, to advertise to the provider.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    /// Look up a tool by the name the model called.
    pub fn get(&self, name: &str) -> Option<std::sync::Arc<dyn Tool>> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .cloned()
    }
}
