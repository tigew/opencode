import { describe, expect, test } from "bun:test"
import {
  decodeToolCalls,
  encodeHistory,
  encodeToolCall,
  encodeToolDeclarations,
  Gemma4StreamParser,
  isGemma4,
  stripThinking,
} from "../../src/provider/gemma4"

describe("gemma4 isGemma4", () => {
  const model = (id: string, apiId = id) => ({ id, api: { id: apiId, url: "", npm: "@ai-sdk/openai-compatible" } }) as any

  test("matches common id forms", () => {
    expect(isGemma4(model("gemma-4"))).toBe(true)
    expect(isGemma4(model("gemma4"))).toBe(true)
    expect(isGemma4(model("google/gemma-4-26b-it"))).toBe(true)
    expect(isGemma4(model("Gemma_4"))).toBe(true)
    expect(isGemma4(model("unsloth/gemma-4-e4b-it-gguf"))).toBe(true)
  })

  test("rejects non-matches", () => {
    expect(isGemma4(model("gemma-3-it"))).toBe(false)
    expect(isGemma4(model("gemma-2"))).toBe(false)
    expect(isGemma4(model("gemini-2.5-pro"))).toBe(false)
  })
})

describe("gemma4 encodeToolDeclarations", () => {
  test("serialises a single-param tool", () => {
    const out = encodeToolDeclarations([
      {
        type: "function",
        function: {
          name: "read",
          description: "Read a file",
          parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
        },
      },
    ])
    expect(out).toContain("<|tool>declaration:read{path:string}<tool|>")
    expect(out).toContain("Read a file")
  })

  test("marks optional params with ?", () => {
    const out = encodeToolDeclarations([
      {
        type: "function",
        function: {
          name: "search",
          parameters: {
            type: "object",
            properties: { q: { type: "string" }, limit: { type: "number" } },
            required: ["q"],
          },
        },
      },
    ])
    expect(out).toContain("q:string")
    expect(out).toContain("limit?:number")
  })

  test("returns empty string when tools are missing", () => {
    expect(encodeToolDeclarations([])).toBe("")
  })
})

describe("gemma4 encodeHistory", () => {
  test("rewrites assistant tool_calls into tokenised text", () => {
    const out = encodeHistory([
      { role: "user", content: "list files" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "list", arguments: '{"path":"."}' } },
        ],
      },
    ])
    expect(out[1].role).toBe("assistant")
    expect(out[1].content).toContain("<|tool_call>call:list{path:<|\"|>.<|\"|>}<tool_call|>")
    expect(out[1].tool_calls).toBeUndefined()
  })

  test("promotes role:tool to user with <|tool_response>", () => {
    const out = encodeHistory([
      {
        role: "assistant",
        content: "",
        tool_calls: [{ id: "call_1", type: "function", function: { name: "list", arguments: "{}" } }],
      },
      { role: "tool", tool_call_id: "call_1", content: "foo\nbar" },
    ])
    expect(out[1].role).toBe("user")
    expect(out[1].content).toContain("<|tool_response>response:list")
    expect(out[1].content).toContain("foo\nbar")
  })

  test("strips leaked thinking from plain assistant text", () => {
    const out = encodeHistory([
      { role: "assistant", content: "hello <|channel>thought secret plan<channel|> world" },
    ])
    expect(out[0].content).toBe("hello  world")
  })
})

describe("gemma4 decodeToolCalls", () => {
  test("parses a single call with a string arg", () => {
    const { content, calls } = decodeToolCalls(
      "ok<|tool_call>call:read{path:<|\"|>./foo.ts<|\"|>}<tool_call|>done",
    )
    expect(content).toBe("okdone")
    expect(calls).toHaveLength(1)
    expect(calls[0].name).toBe("read")
    expect(calls[0].args).toEqual({ path: "./foo.ts" })
  })

  test("parses mixed types (string, number, bool, null)", () => {
    const { calls } = decodeToolCalls(
      "<|tool_call>call:do{a:<|\"|>hi<|\"|>,b:42,c:true,d:null}<tool_call|>",
    )
    expect(calls[0].args).toEqual({ a: "hi", b: 42, c: true, d: null })
  })

  test("handles <|\"|>-wrapped values containing braces and quotes", () => {
    const { calls } = decodeToolCalls(
      "<|tool_call>call:write{body:<|\"|>{\"x\":1,\"y\":\"a,b\"}<|\"|>}<tool_call|>",
    )
    expect(calls[0].args.body).toEqual({ x: 1, y: "a,b" })
  })

  test("leaves content untouched when no markers present", () => {
    const { content, calls } = decodeToolCalls("plain text, no tools")
    expect(content).toBe("plain text, no tools")
    expect(calls).toHaveLength(0)
  })
})

describe("gemma4 stripThinking", () => {
  test("removes thinking blocks", () => {
    expect(stripThinking("a<|channel>thought b<channel|>c")).toBe("ac")
  })
  test("is a no-op on inputs with no marker", () => {
    expect(stripThinking("plain")).toBe("plain")
  })
  test("drops unterminated thinking to end-of-text", () => {
    expect(stripThinking("a<|channel>thought dangling")).toBe("a")
  })
})

describe("gemma4 Gemma4StreamParser", () => {
  test("streams plain text verbatim", () => {
    const p = new Gemma4StreamParser()
    const a = p.push("hello ")
    const b = p.push("world")
    const c = p.flush()
    expect((a.content + b.content + c.content)).toBe("hello world")
    expect([...a.calls, ...b.calls, ...c.calls]).toHaveLength(0)
  })

  test("extracts a tool call even when split across chunks", () => {
    const p = new Gemma4StreamParser()
    const chunks = [
      "pre ",
      "<|tool_",
      "call>call:read",
      "{path:<|",
      "\"|>./a<|\"|>}<tool_call",
      "|> post",
    ]
    let content = ""
    const calls: any[] = []
    for (const c of chunks) {
      const r = p.push(c)
      content += r.content
      calls.push(...r.calls)
    }
    const tail = p.flush()
    content += tail.content
    calls.push(...tail.calls)
    expect(content.trim()).toBe("pre  post".trim())
    expect(calls).toHaveLength(1)
    expect(calls[0].name).toBe("read")
    expect(calls[0].args).toEqual({ path: "./a" })
  })

  test("strips thinking segments across chunk boundaries", () => {
    const p = new Gemma4StreamParser()
    const chunks = ["hi ", "<|channel>thought", " should not leak ", "<channel|>", "bye"]
    let content = ""
    for (const c of chunks) content += p.push(c).content
    content += p.flush().content
    expect(content).toBe("hi bye")
  })
})

describe("gemma4 encodeToolCall", () => {
  test("escapes string args with <|\"|>", () => {
    expect(encodeToolCall("read", '{"path":"./x"}'))
      .toBe("<|tool_call>call:read{path:<|\"|>./x<|\"|>}<tool_call|>")
  })
})
