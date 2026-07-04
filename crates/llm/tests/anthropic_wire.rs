//! Wire-format tests for the Anthropic provider, using a mocked HTTP server.
//!
//! These verify that `AnthropicProvider::turn` builds the `/v1/messages` request
//! correctly (auth headers, body shape) and maps the response back into the
//! common types — without touching the real API or needing a key.

#![cfg(feature = "anthropic")]

use llm::provider::{LlmProvider, TurnRequest};
use llm::types::{ContentBlock, Message, Role, StopReason, ToolDefinition};
use llm::{ProviderConfig, ProviderKind};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn provider_pointing_at(server: &MockServer) -> llm::providers::anthropic::AnthropicProvider {
    let config = ProviderConfig {
        kind: ProviderKind::Anthropic,
        model: "claude-opus-4-8".to_string(),
        api_key: Some("test-key".to_string()),
        base_url: Some(server.uri()),
        max_tokens: 1024,
    };
    llm::providers::anthropic::AnthropicProvider::new(config).unwrap()
}

#[tokio::test]
async fn turn_sends_correct_request_and_parses_text_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "text", "text": "Hello there." }],
            "stop_reason": "end_turn"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_pointing_at(&server);
    let request = TurnRequest {
        messages: vec![Message::user_text("hi")],
        system: Some("be brief".to_string()),
        tools: vec![],
        max_tokens: 0, // 0 → provider falls back to config default
    };

    let resp = provider.turn(&request).await.unwrap();
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert_eq!(resp.blocks.len(), 1);
    match &resp.blocks[0] {
        ContentBlock::Text { text } => assert_eq!(text, "Hello there."),
        other => panic!("expected text block, got {other:?}"),
    }
}

#[tokio::test]
async fn turn_maps_tool_use_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "content": [
                { "type": "text", "text": "Let me check." },
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "lookup",
                    "input": { "q": "X" }
                }
            ],
            "stop_reason": "tool_use"
        })))
        .mount(&server)
        .await;

    let provider = provider_pointing_at(&server);
    let request = TurnRequest {
        messages: vec![Message::user_text("did we have X?")],
        system: None,
        tools: vec![ToolDefinition {
            name: "lookup".to_string(),
            description: "look up".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }],
        max_tokens: 256,
    };

    let resp = provider.turn(&request).await.unwrap();
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    assert_eq!(resp.blocks.len(), 2);
    match &resp.blocks[1] {
        ContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "toolu_abc");
            assert_eq!(name, "lookup");
            assert_eq!(input, &serde_json::json!({"q": "X"}));
        }
        other => panic!("expected tool_use block, got {other:?}"),
    }
}

#[tokio::test]
async fn turn_sends_required_fields_in_body() {
    let server = MockServer::start().await;

    // Capture and assert the request body shape.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(|req: &Request| {
            let body: serde_json::Value = req.body_json().unwrap();
            // max_tokens is required by Anthropic.
            assert!(body.get("max_tokens").is_some(), "max_tokens missing");
            assert_eq!(body["model"], "claude-opus-4-8");
            assert_eq!(body["system"], "be brief");
            // tool definition uses `input_schema` (Anthropic shape).
            assert_eq!(body["tools"][0]["name"], "lookup");
            assert!(body["tools"][0].get("input_schema").is_some());
            // messages carry the user turn with a text content block.
            assert_eq!(body["messages"][0]["role"], "user");
            assert_eq!(body["messages"][0]["content"][0]["type"], "text");
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{ "type": "text", "text": "ok" }],
                "stop_reason": "end_turn"
            }))
        })
        .mount(&server)
        .await;

    let provider = provider_pointing_at(&server);
    let request = TurnRequest {
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "hi".to_string(),
            }],
        }],
        system: Some("be brief".to_string()),
        tools: vec![ToolDefinition {
            name: "lookup".to_string(),
            description: "look up".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }],
        max_tokens: 512,
    };

    let resp = provider.turn(&request).await.unwrap();
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn turn_surfaces_api_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
            "type": "error",
            "error": { "type": "authentication_error", "message": "invalid key" }
        })))
        .mount(&server)
        .await;

    let provider = provider_pointing_at(&server);
    let request = TurnRequest {
        messages: vec![Message::user_text("hi")],
        system: None,
        tools: vec![],
        max_tokens: 64,
    };

    let err = provider.turn(&request).await.unwrap_err();
    assert!(
        matches!(err, llm::LlmError::Api(_)),
        "expected Api error, got {err:?}"
    );
}
