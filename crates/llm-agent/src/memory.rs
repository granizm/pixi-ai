//! Conversation memory — the seam for future history compaction.
//!
//! Today this is a thin wrapper over `Vec<Message>` that simply retains every
//! turn (the phase-1 "agent holds the history" model). It exists as its own type
//! so that a later phase can swap in a compacting implementation — periodically
//! summarising old turns into a memory note and dropping them before the context
//! window fills — **without changing the agent or any caller**. The public
//! surface here is deliberately the minimum the agent needs.

use llm::Message;

/// Holds the running conversation history for an [`crate::agent::LlmAgent`].
///
/// Phase 1: retains all turns verbatim. Phase 3 (later): compaction — summarise
/// and evict old turns while preserving recent context. Callers depend only on
/// the methods below, so that upgrade is transparent.
#[derive(Debug, Default, Clone)]
pub struct ConversationMemory {
    history: Vec<Message>,
}

impl ConversationMemory {
    /// New, empty conversation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a turn (user or assistant).
    pub fn push(&mut self, message: Message) {
        self.history.push(message);
    }

    /// The full message history to send on the next provider turn.
    ///
    /// Phase 3 will return a compacted view here (summary + recent turns); the
    /// agent loop is written against this method precisely so that change is
    /// invisible to it.
    pub fn messages(&self) -> Vec<Message> {
        self.history.clone()
    }

    /// Clear the conversation (start a new topic).
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Number of retained turns.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}
