import type * as Provider from "./provider"

// Gemma 4 dialect: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
//
// Gemma 4 speaks a non-standard wire format when tool-calling. It does not use
// OpenAI's JSON `tools` / `tool_calls` structure. Instead:
//
//   declaration: <|tool>declaration:name{param:type,…}<tool|>         (in system prompt)
//   call:        <|tool_call>call:name{param:<|"|>value<|"|>}<tool_call|>
//   response:    <|tool_response>response:name{field:value}<tool_response|>
//   thinking:    <|channel>thought … <channel|>                        (strip from history)
//
// When served locally via llama.cpp / Ollama / LM Studio with `@ai-sdk/openai-compatible`,
// these tokens flow as raw text in `choices[0].delta.content`. The helpers here encode
// outgoing history into Gemma's dialect and decode incoming text into OpenAI-style
// `tool_calls` so the rest of opencode can stay format-agnostic.

export const GEMMA4_STRING_DELIM = "<|\"|>"
const THOUGHT_OPEN = "<|channel>thought"
const THOUGHT_CLOSE = "<channel|>"

export function isGemma4(model: Pick<Provider.Model, "id" | "api">): boolean {
  const re = /gemma[-_.]?4/i
  return re.test(model.id) || re.test(model.api?.id ?? "")
}

type OpenAIMessage = {
  role: "system" | "user" | "assistant" | "tool" | string
  content?: string | null | Array<any>
  tool_calls?: Array<{ id?: string; type?: string; function?: { name?: string; arguments?: string } }>
  tool_call_id?: string
  name?: string
}

type OpenAITool = {
  type?: string
  function?: {
    name?: string
    description?: string
    parameters?: {
      type?: string
      properties?: Record<string, { type?: string; description?: string }>
      required?: string[]
    }
  }
}

type ParsedToolCall = {
  id: string
  name: string
  args: Record<string, unknown>
}

// Serialise an OpenAI tool schema into a Gemma declaration block to be prepended
// to the system message. Non-object parameters are stringified as JSON for forward
// compat; in practice all tools in opencode use object parameter schemas.
export function encodeToolDeclarations(tools: OpenAITool[]): string {
  if (!tools?.length) return ""
  const blocks = tools
    .map((tool) => {
      const fn = tool.function
      if (!fn?.name) return ""
      const props = fn.parameters?.properties ?? {}
      const required = new Set(fn.parameters?.required ?? [])
      const params = Object.entries(props)
        .map(([name, spec]) => {
          const type = spec?.type ?? "string"
          const marker = required.has(name) ? "" : "?"
          return `${name}${marker}:${type}`
        })
        .join(",")
      const description = fn.description ? ` ${fn.description}` : ""
      return `<|tool>declaration:${fn.name}{${params}}<tool|>${description}`
    })
    .filter(Boolean)
  return blocks.join("\n")
}

// Serialise a previously-executed tool call into Gemma's `<|tool_call>` text form
// so it can be embedded in an assistant message.
export function encodeToolCall(name: string, argumentsJson: string): string {
  let parsed: Record<string, unknown> = {}
  try {
    const raw = argumentsJson ? JSON.parse(argumentsJson) : {}
    if (raw && typeof raw === "object" && !Array.isArray(raw)) parsed = raw as Record<string, unknown>
  } catch {
    // fall through with empty object
  }
  const body = Object.entries(parsed)
    .map(([k, v]) => `${k}:${encodeValue(v)}`)
    .join(",")
  return `<|tool_call>call:${name}{${body}}<tool_call|>`
}

// Serialise a tool result. Gemma has no `tool` role: the response rides as a
// user-turn text block.
export function encodeToolResponse(name: string, content: string): string {
  const value = encodeValue(content)
  return `<|tool_response>response:${name}{result:${value}}<tool_response|>`
}

function encodeValue(v: unknown): string {
  if (typeof v === "string") return `${GEMMA4_STRING_DELIM}${v}${GEMMA4_STRING_DELIM}`
  if (typeof v === "number" || typeof v === "boolean") return String(v)
  if (v === null || v === undefined) return "null"
  // Objects / arrays: serialise as JSON inside the string delimiter so the
  // model sees a single opaque value. Gemma 4 handles nested structures this
  // way in practice.
  return `${GEMMA4_STRING_DELIM}${JSON.stringify(v).replaceAll(GEMMA4_STRING_DELIM, "")}${GEMMA4_STRING_DELIM}`
}

// Rewrite a conversation history from OpenAI shape into the Gemma dialect. Any
// assistant message with `tool_calls` becomes a plain assistant text message
// carrying the `<|tool_call>…<tool_call|>` tokens. Any `role:"tool"` message is
// promoted to `role:"user"` carrying `<|tool_response>…<tool_response|>`.
//
// We also index tool_call_id → name so we can name the response correctly.
export function encodeHistory(messages: OpenAIMessage[]): OpenAIMessage[] {
  const nameById = new Map<string, string>()
  for (const msg of messages) {
    if (msg.role === "assistant" && Array.isArray(msg.tool_calls)) {
      for (const tc of msg.tool_calls) {
        if (tc.id && tc.function?.name) nameById.set(tc.id, tc.function.name)
      }
    }
  }

  const out: OpenAIMessage[] = []
  for (const msg of messages) {
    if (msg.role === "assistant" && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
      const text = typeof msg.content === "string" ? msg.content : ""
      const calls = msg.tool_calls
        .map((tc) => encodeToolCall(tc.function?.name ?? "", tc.function?.arguments ?? "{}"))
        .join("")
      out.push({ role: "assistant", content: `${text}${calls}` })
      continue
    }

    if (msg.role === "tool") {
      const name = (msg.tool_call_id && nameById.get(msg.tool_call_id)) || msg.name || "tool"
      const body = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content ?? "")
      out.push({ role: "user", content: encodeToolResponse(name, body) })
      continue
    }

    if (msg.role === "assistant" && typeof msg.content === "string") {
      // Never replay leaked thinking back to the model.
      out.push({ ...msg, content: stripThinking(msg.content) })
      continue
    }

    out.push(msg)
  }
  return out
}

// Remove any `<|channel>thought … <channel|>` segments from a completed text
// block. Safe on inputs with no markers (returns input unchanged). For streamed
// text, use `ThinkingStripper` below which handles chunk boundaries.
export function stripThinking(text: string): string {
  if (!text || !text.includes(THOUGHT_OPEN)) return text
  let out = ""
  let i = 0
  while (i < text.length) {
    const open = text.indexOf(THOUGHT_OPEN, i)
    if (open === -1) {
      out += text.slice(i)
      break
    }
    out += text.slice(i, open)
    const close = text.indexOf(THOUGHT_CLOSE, open + THOUGHT_OPEN.length)
    if (close === -1) break // unterminated; drop the rest
    i = close + THOUGHT_CLOSE.length
  }
  return out
}

// Parse all `<|tool_call>call:name{…}<tool_call|>` occurrences out of a text
// block. The plain text (with tool-call markers removed) is returned alongside
// the structured calls.
export function decodeToolCalls(text: string): { content: string; calls: ParsedToolCall[] } {
  const calls: ParsedToolCall[] = []
  if (!text || !text.includes("<|tool_call>")) return { content: text, calls }

  let content = ""
  let i = 0
  while (i < text.length) {
    const open = text.indexOf("<|tool_call>", i)
    if (open === -1) {
      content += text.slice(i)
      break
    }
    content += text.slice(i, open)
    const close = text.indexOf("<tool_call|>", open)
    if (close === -1) {
      // Unterminated tool call — leave the rest of the text in place; the
      // stream parser will retry once more bytes arrive.
      content += text.slice(open)
      break
    }
    const inner = text.slice(open + "<|tool_call>".length, close)
    const parsed = parseCallBody(inner)
    if (parsed) calls.push(parsed)
    i = close + "<tool_call|>".length
  }
  return { content, calls }
}

function parseCallBody(inner: string): ParsedToolCall | null {
  // Expected shape: `call:<name>{<k>:<v>,<k>:<v>}`
  const callPrefix = "call:"
  const start = inner.indexOf(callPrefix)
  if (start === -1) return null
  const braceOpen = inner.indexOf("{", start + callPrefix.length)
  if (braceOpen === -1) return null
  const name = inner.slice(start + callPrefix.length, braceOpen).trim()
  if (!name) return null
  const braceClose = findMatchingBrace(inner, braceOpen)
  if (braceClose === -1) return null
  const args = parseArgs(inner.slice(braceOpen + 1, braceClose))
  return { id: synthId(), name, args }
}

// Find the closing `}` for the `{` at `openIdx`, respecting the `<|"|>` string
// delimiter (so `{` / `}` inside a string are not miscounted). Returns -1 if no
// balanced close exists.
function findMatchingBrace(s: string, openIdx: number): number {
  let depth = 0
  let i = openIdx
  while (i < s.length) {
    if (s.startsWith(GEMMA4_STRING_DELIM, i)) {
      const next = s.indexOf(GEMMA4_STRING_DELIM, i + GEMMA4_STRING_DELIM.length)
      if (next === -1) return -1
      i = next + GEMMA4_STRING_DELIM.length
      continue
    }
    const ch = s[i]
    if (ch === "{") depth++
    else if (ch === "}") {
      depth--
      if (depth === 0) return i
    }
    i++
  }
  return -1
}

function parseArgs(body: string): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  let i = 0
  while (i < body.length) {
    // skip whitespace and separators
    while (i < body.length && (body[i] === "," || body[i] === " " || body[i] === "\n" || body[i] === "\t")) i++
    if (i >= body.length) break
    const colon = body.indexOf(":", i)
    if (colon === -1) break
    const key = body.slice(i, colon).trim()
    let j = colon + 1
    while (j < body.length && (body[j] === " " || body[j] === "\t")) j++
    const [value, end] = readValue(body, j)
    out[key] = value
    i = end
    while (i < body.length && body[i] !== ",") i++
  }
  return out
}

function readValue(s: string, start: number): [unknown, number] {
  if (s.startsWith(GEMMA4_STRING_DELIM, start)) {
    const from = start + GEMMA4_STRING_DELIM.length
    const to = s.indexOf(GEMMA4_STRING_DELIM, from)
    if (to === -1) return [s.slice(from), s.length]
    const raw = s.slice(from, to)
    // Try to recover JSON-ish objects/arrays that were wrapped as strings.
    const trimmed = raw.trim()
    if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
      try {
        return [JSON.parse(trimmed), to + GEMMA4_STRING_DELIM.length]
      } catch {
        // fall through
      }
    }
    return [raw, to + GEMMA4_STRING_DELIM.length]
  }
  if (s.startsWith("{", start)) {
    const end = findMatchingBrace(s, start)
    if (end === -1) return [s.slice(start), s.length]
    return [parseArgs(s.slice(start + 1, end)), end + 1]
  }
  // bare token: number, bool, null, or bareword — read up to `,` or end
  let end = start
  while (end < s.length && s[end] !== "," && s[end] !== "}") end++
  const token = s.slice(start, end).trim()
  if (token === "true") return [true, end]
  if (token === "false") return [false, end]
  if (token === "null") return [null, end]
  const num = Number(token)
  if (!Number.isNaN(num) && token !== "") return [num, end]
  return [token, end]
}

// Stateful streaming parser for a sequence of text chunks. Emits OpenAI-style
// `tool_calls` deltas when a complete `<|tool_call>…<tool_call|>` is observed,
// strips `<|channel>thought…<channel|>` segments, and emits text between /
// after them. Holds back bytes that might be the start of a marker until enough
// bytes have arrived to decide.
export class Gemma4StreamParser {
  private buffer = ""
  private inThought = false
  private readonly maxHoldback: number

  constructor() {
    // Longest open marker is `<|channel>thought` (17). Hold back at most 24
    // bytes so a split marker is never emitted as text.
    this.maxHoldback = 24
  }

  // Feed a new text chunk, returns {content, calls} produced by this chunk.
  push(chunk: string): { content: string; calls: ParsedToolCall[] } {
    this.buffer += chunk
    return this.drain(false)
  }

  // Call at end-of-stream to flush any remaining safe text.
  flush(): { content: string; calls: ParsedToolCall[] } {
    return this.drain(true)
  }

  private drain(final: boolean): { content: string; calls: ParsedToolCall[] } {
    const calls: ParsedToolCall[] = []
    let content = ""

    while (this.buffer.length > 0) {
      if (this.inThought) {
        const close = this.buffer.indexOf(THOUGHT_CLOSE)
        if (close === -1) {
          // wait for more
          if (final) this.buffer = ""
          break
        }
        this.buffer = this.buffer.slice(close + THOUGHT_CLOSE.length)
        this.inThought = false
        continue
      }

      const thoughtOpen = this.buffer.indexOf(THOUGHT_OPEN)
      const callOpen = this.buffer.indexOf("<|tool_call>")

      // Figure out the nearest structural marker.
      const next = [thoughtOpen, callOpen].filter((v) => v !== -1).sort((a, b) => a - b)[0] ?? -1

      if (next === -1) {
        // No complete marker. Emit all but the last `maxHoldback` bytes (which
        // might be the start of a future marker).
        const safeEnd = final ? this.buffer.length : Math.max(0, this.buffer.length - this.maxHoldback)
        content += this.buffer.slice(0, safeEnd)
        this.buffer = this.buffer.slice(safeEnd)
        break
      }

      content += this.buffer.slice(0, next)
      this.buffer = this.buffer.slice(next)

      if (this.buffer.startsWith(THOUGHT_OPEN)) {
        this.buffer = this.buffer.slice(THOUGHT_OPEN.length)
        this.inThought = true
        continue
      }

      // tool_call
      const closeIdx = this.buffer.indexOf("<tool_call|>")
      if (closeIdx === -1) {
        // Incomplete tool call — wait for more bytes.
        if (final) {
          // Drop the partial; it's unusable.
          this.buffer = ""
        }
        break
      }
      const inner = this.buffer.slice("<|tool_call>".length, closeIdx)
      const parsed = parseCallBody(inner)
      if (parsed) calls.push(parsed)
      this.buffer = this.buffer.slice(closeIdx + "<tool_call|>".length)
    }

    return { content, calls }
  }
}

let synthCounter = 0
function synthId(): string {
  synthCounter = (synthCounter + 1) % 1_000_000
  return `call_gemma4_${Date.now().toString(36)}_${synthCounter}`
}
