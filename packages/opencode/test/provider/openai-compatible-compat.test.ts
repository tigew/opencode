import { describe, expect, test } from "bun:test"
import {
  getOpenAICompatibleToolParsers,
  rewriteOpenAICompatibleJsonResponse,
  rewriteOpenAICompatibleRequestBody,
  rewriteOpenAICompatibleStreamResponse,
} from "../../src/provider/openai-compatible-compat"

describe("openai-compatible compat: parser registration", () => {
  test("ignores malformed entries", () => {
    expect(getOpenAICompatibleToolParsers({ toolParser: null })).toEqual([])
    expect(getOpenAICompatibleToolParsers({ toolParser: [1, "x", {}] })).toEqual([])
  })
  test("keeps known parser types", () => {
    expect(
      getOpenAICompatibleToolParsers({
        toolParser: [{ type: "gemma4" }, { type: "raw-function-call" }, { type: "json" }],
      }),
    ).toEqual([{ type: "gemma4" }, { type: "raw-function-call" }, { type: "json" }])
  })
})

describe("openai-compatible compat: raw-function-call request", () => {
  test("rewrites tools and tool_choice to legacy function_call", () => {
    const rewritten = rewriteOpenAICompatibleRequestBody(
      {
        model: "demo",
        tools: [
          {
            type: "function",
            function: {
              name: "bash",
              description: "Run a shell command",
              parameters: {
                type: "object",
                properties: { command: { type: "string" } },
                required: ["command"],
              },
            },
          },
        ],
        tool_choice: { type: "function", function: { name: "bash" } },
        parallel_tool_calls: false,
      },
      [{ type: "raw-function-call" }],
    )
    expect(rewritten.tools).toBeUndefined()
    expect(rewritten.tool_choice).toBeUndefined()
    expect(rewritten.functions).toHaveLength(1)
    expect(rewritten.function_call).toEqual({ name: "bash" })
  })
})

describe("openai-compatible compat: raw-function-call json response", () => {
  test("rewrites function_call → tool_calls", () => {
    const rewritten = rewriteOpenAICompatibleJsonResponse(
      {
        choices: [
          {
            message: {
              role: "assistant",
              content: null,
              function_call: { name: "bash", arguments: '{"command":"ls -la"}' },
            },
            finish_reason: "function_call",
          },
        ],
      },
      [{ type: "raw-function-call" }],
    )
    expect(rewritten.choices[0].finish_reason).toBe("tool_calls")
    expect(rewritten.choices[0].message.function_call).toBeUndefined()
    expect(rewritten.choices[0].message.tool_calls[0].function.name).toBe("bash")
  })
})

describe("openai-compatible compat: gemma4 request", () => {
  test("injects declaration block and drops tools", () => {
    const rewritten = rewriteOpenAICompatibleRequestBody(
      {
        messages: [
          { role: "system", content: "You are helpful." },
          { role: "user", content: "hi" },
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "read",
              parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
            },
          },
        ],
        tool_choice: "auto",
      },
      [{ type: "gemma4" }],
    )
    expect(rewritten.tools).toBeUndefined()
    expect(rewritten.tool_choice).toBeUndefined()
    expect(rewritten.messages[0].content).toContain("<|tool>declaration:read{path:string}<tool|>")
    expect(rewritten.messages[0].content).toContain("You are helpful.")
  })

  test("rewrites prior tool_calls and role:tool history", () => {
    const rewritten = rewriteOpenAICompatibleRequestBody(
      {
        messages: [
          { role: "user", content: "x" },
          {
            role: "assistant",
            content: "",
            tool_calls: [
              { id: "c1", type: "function", function: { name: "read", arguments: '{"path":"./a"}' } },
            ],
          },
          { role: "tool", tool_call_id: "c1", content: "hello" },
        ],
      },
      [{ type: "gemma4" }],
    )
    expect(rewritten.messages[1].tool_calls).toBeUndefined()
    expect(rewritten.messages[1].content).toContain("<|tool_call>call:read")
    expect(rewritten.messages[2].role).toBe("user")
    expect(rewritten.messages[2].content).toContain("<|tool_response>response:read")
  })
})

describe("openai-compatible compat: gemma4 json response", () => {
  test("extracts tool calls from content", () => {
    const rewritten = rewriteOpenAICompatibleJsonResponse(
      {
        choices: [
          {
            message: {
              role: "assistant",
              content:
                "ok<|tool_call>call:read{path:<|\"|>./foo.ts<|\"|>}<tool_call|>",
            },
            finish_reason: "stop",
          },
        ],
      },
      [{ type: "gemma4" }],
    )
    expect(rewritten.choices[0].finish_reason).toBe("tool_calls")
    expect(rewritten.choices[0].message.content).toBe("ok")
    expect(rewritten.choices[0].message.tool_calls).toHaveLength(1)
    expect(rewritten.choices[0].message.tool_calls[0].function.name).toBe("read")
    expect(JSON.parse(rewritten.choices[0].message.tool_calls[0].function.arguments)).toEqual({ path: "./foo.ts" })
  })

  test("strips thinking channel from content", () => {
    const rewritten = rewriteOpenAICompatibleJsonResponse(
      {
        choices: [
          {
            message: {
              role: "assistant",
              content: "hello <|channel>thought secret<channel|> world",
            },
            finish_reason: "stop",
          },
        ],
      },
      [{ type: "gemma4" }],
    )
    expect(rewritten.choices[0].message.content).not.toContain("secret")
    expect(rewritten.choices[0].message.content).toContain("hello")
    expect(rewritten.choices[0].message.content).toContain("world")
  })
})

describe("openai-compatible compat: gemma4 sse response", () => {
  test("converts streamed <|tool_call> tokens into synthetic tool_calls delta", () => {
    const input = [
      'data: {"id":"1","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"I will read."},"finish_reason":null}]}',
      'data: {"id":"1","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"<|tool_call>call:read{path:<|\\"|>./foo<|\\"|>}<tool_call|>"},"finish_reason":null}]}',
      'data: {"id":"1","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
      "data: [DONE]",
    ].join("\n\n")

    const out = rewriteOpenAICompatibleStreamResponse(input, [{ type: "gemma4" }])
    expect(out).toContain("tool_calls")
    expect(out).toContain("\"name\":\"read\"")
    expect(out).toContain("\"finish_reason\":\"tool_calls\"")
  })

  test("drops streamed thinking-channel content", () => {
    const input = [
      'data: {"id":"1","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi <|channel>thought"},"finish_reason":null}]}',
      'data: {"id":"1","created":1,"model":"m","choices":[{"index":0,"delta":{"content":" plan plan plan<channel|> bye"},"finish_reason":null}]}',
      'data: {"id":"1","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
      "data: [DONE]",
    ].join("\n\n")

    const out = rewriteOpenAICompatibleStreamResponse(input, [{ type: "gemma4" }])
    expect(out).not.toContain("plan plan plan")
    expect(out).toContain("hi")
    expect(out).toContain("bye")
  })

  test("passes through cleanly when stream has no gemma markers", () => {
    const input = [
      'data: {"id":"1","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"regular answer"},"finish_reason":"stop"}]}',
      "data: [DONE]",
    ].join("\n\n")
    const out = rewriteOpenAICompatibleStreamResponse(input, [{ type: "gemma4" }])
    expect(out).toContain("regular answer")
    expect(out).toContain("[DONE]")
  })
})
