import {
  decodeToolCalls,
  encodeHistory,
  encodeToolDeclarations,
  Gemma4StreamParser,
  stripThinking,
} from "./gemma4"

// Compatibility shim that rewrites requests and responses at the HTTP boundary
// for openai-compatible endpoints serving models whose tool-calling conventions
// diverge from OpenAI's `tools` / `tool_calls` JSON shape. Activated per-provider
// via `options.toolParser`; the provider fetch wrapper calls into these helpers
// when any parsers are configured.
//
// Mirrors the shape of opencode PR #16531 so a future upstream merge is clean,
// with an added `gemma4` parser for Gemma 4's `<|tool_call>` token dialect.

type RawFunctionCallToolParser = {
  type: "raw-function-call"
}

type JsonToolParser = {
  type: "json"
}

type Gemma4ToolParser = {
  type: "gemma4"
}

export type OpenAICompatibleToolParser = RawFunctionCallToolParser | JsonToolParser | Gemma4ToolParser

type OpenAICompatibleToolCall = {
  id: string
  type: "function"
  function: {
    name: string
    arguments: string
  }
}

const SYNTHETIC_TOOL_CALL_ID = "call_opencode_compat_0"

export function getOpenAICompatibleToolParsers(options: Record<string, any>): OpenAICompatibleToolParser[] {
  const input = options["toolParser"]
  if (!Array.isArray(input)) return []

  return input.flatMap((item): OpenAICompatibleToolParser[] => {
    if (!item || typeof item !== "object") return []
    if (item.type === "raw-function-call") return [{ type: "raw-function-call" }]
    if (item.type === "json") return [{ type: "json" }]
    if (item.type === "gemma4") return [{ type: "gemma4" }]
    return []
  })
}

export function rewriteOpenAICompatibleRequestBody(
  body: Record<string, any>,
  parsers: OpenAICompatibleToolParser[],
): Record<string, any> {
  let next = body
  if (parsers.some((p) => p.type === "gemma4")) next = applyGemma4Request(next)
  if (parsers.some((p) => p.type === "raw-function-call")) next = applyRawFunctionCallRequest(next)
  return next
}

function applyGemma4Request(body: Record<string, any>): Record<string, any> {
  const next = JSON.parse(JSON.stringify(body))
  const tools = Array.isArray(next.tools) ? next.tools : []
  const declarations = encodeToolDeclarations(tools)

  if (declarations) {
    const messages = Array.isArray(next.messages) ? next.messages : []
    const firstSystem = messages.findIndex((m: any) => m?.role === "system")
    const block = `\n${declarations}`
    if (firstSystem >= 0) {
      const existing = messages[firstSystem]
      const content = typeof existing.content === "string" ? existing.content : ""
      messages[firstSystem] = { ...existing, content: `${content}${block}` }
    } else {
      messages.unshift({ role: "system", content: declarations })
    }
    next.messages = messages
  }

  if (Array.isArray(next.messages)) {
    next.messages = encodeHistory(next.messages)
  }

  delete next.tools
  delete next.tool_choice
  delete next.parallel_tool_calls
  return next
}

function applyRawFunctionCallRequest(body: Record<string, any>): Record<string, any> {
  const next = JSON.parse(JSON.stringify(body))
  const tools = Array.isArray(next.tools) ? next.tools : []
  const functions = tools
    .filter((tool: any) => tool?.type === "function" && tool.function)
    .map((tool: any) => tool.function)

  if (functions.length > 0 && next.functions === undefined) {
    next.functions = functions
  }

  if (next.function_call === undefined && next.tool_choice !== undefined) {
    if (next.tool_choice === "auto" || next.tool_choice === "none") {
      next.function_call = next.tool_choice
    } else if (next.tool_choice === "required") {
      next.function_call = functions.length === 1 ? { name: functions[0].name } : "auto"
    } else if (typeof next.tool_choice === "object" && next.tool_choice?.function?.name) {
      next.function_call = { name: next.tool_choice.function.name }
    }
  }

  delete next.tools
  delete next.tool_choice
  delete next.parallel_tool_calls
  return next
}

export function rewriteOpenAICompatibleJsonResponse(
  body: Record<string, any>,
  parsers: OpenAICompatibleToolParser[],
): Record<string, any> {
  const next = JSON.parse(JSON.stringify(body))
  const choice = next?.choices?.[0]
  if (!choice) return next

  normalizeLegacyFunctionCall(choice)

  if (parsers.some((p) => p.type === "gemma4") && !choice.message?.tool_calls?.length) {
    applyGemma4ToMessage(choice)
  }

  if (!choice.message?.tool_calls?.length) {
    const parsed = parseToolCallFromContent(choice.message?.content, parsers)
    if (parsed) {
      choice.message.content = null
      choice.message.tool_calls = [parsed]
      choice.finish_reason = "tool_calls"
    }
  }

  return next
}

function applyGemma4ToMessage(choice: Record<string, any>): void {
  const content = choice.message?.content
  if (typeof content !== "string" || content.length === 0) return
  const stripped = stripThinking(content)
  const { content: remaining, calls } = decodeToolCalls(stripped)
  if (calls.length > 0) {
    choice.message.content = remaining.trim() || null
    choice.message.tool_calls = calls.map((c) => ({
      id: c.id,
      type: "function" as const,
      function: { name: c.name, arguments: JSON.stringify(c.args) },
    }))
    choice.finish_reason = "tool_calls"
  } else if (stripped !== content) {
    choice.message.content = stripped
  }
}

export function rewriteOpenAICompatibleStreamResponse(text: string, parsers: OpenAICompatibleToolParser[]): string {
  const events = parseSSEEvents(text)
  let transformed = events.map((event) => transformLegacyChunkEvent(event))

  if (parsers.some((p) => p.type === "gemma4")) {
    transformed = applyGemma4ToStream(transformed)
  }

  if (!parsers.some((parser) => parser.type === "json")) {
    return serializeSSEEvents(transformed)
  }

  const content = transformed
    .flatMap((event) => {
      if (event.type !== "json") return []
      const choice = event.value?.choices?.[0]
      const delta = choice?.delta
      return typeof delta?.content === "string" ? [delta.content] : []
    })
    .join("")
    .trim()

  const alreadyHasToolCalls = transformed.some((event) => {
    if (event.type !== "json") return false
    const choice = event.value?.choices?.[0]
    return Boolean(choice?.delta?.tool_calls?.length)
  })

  if (alreadyHasToolCalls) {
    return serializeSSEEvents(transformed)
  }

  const parsed = parseToolCallFromContent(content, parsers)
  if (!parsed) return serializeSSEEvents(transformed)

  const firstJson = transformed.find(
    (event): event is Extract<ParsedSSEEvent, { type: "json" }> => event.type === "json",
  )
  const usage = [...transformed]
    .reverse()
    .find((event): event is Extract<ParsedSSEEvent, { type: "json" }> => event.type === "json" && !!event.value?.usage)
    ?.value?.usage

  const synthetic: ParsedSSEEvent[] = []
  if (firstJson) {
    synthetic.push({
      type: "json",
      value: {
        id: firstJson.value?.id,
        created: firstJson.value?.created,
        model: firstJson.value?.model,
        choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
      },
    })
    synthetic.push({
      type: "json",
      value: {
        id: firstJson.value?.id,
        created: firstJson.value?.created,
        model: firstJson.value?.model,
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              tool_calls: [{ index: 0, id: parsed.id, type: "function", function: parsed.function }],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage,
      },
    })
  }

  if (transformed.some((event) => event.type === "done")) synthetic.push({ type: "done" })
  return serializeSSEEvents(synthetic)
}

function applyGemma4ToStream(events: ParsedSSEEvent[]): ParsedSSEEvent[] {
  const parser = new Gemma4StreamParser()
  const out: ParsedSSEEvent[] = []
  let toolIndex = 0
  let template: Record<string, any> | undefined

  for (const event of events) {
    if (event.type !== "json") {
      out.push(event)
      continue
    }
    const value = JSON.parse(JSON.stringify(event.value))
    const choice = value?.choices?.[0]
    template ??= { id: value?.id, created: value?.created, model: value?.model }

    if (!choice) {
      out.push({ type: "json", value })
      continue
    }

    const delta = choice.delta
    if (typeof delta?.content !== "string" || delta.content.length === 0) {
      out.push({ type: "json", value })
      continue
    }

    const { content, calls } = parser.push(delta.content)
    if (content) {
      delta.content = content
      out.push({ type: "json", value })
    } else {
      // Preserve role-only / empty-delta boundary chunks, but drop chunks whose
      // content was entirely consumed by thinking/tool markers.
      delta.content = ""
      if (delta.role || delta.tool_calls) out.push({ type: "json", value })
    }

    for (const call of calls) {
      out.push({
        type: "json",
        value: {
          ...template,
          choices: [
            {
              index: 0,
              delta: {
                role: "assistant",
                tool_calls: [
                  {
                    index: toolIndex++,
                    id: call.id,
                    type: "function",
                    function: { name: call.name, arguments: JSON.stringify(call.args) },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
      })
    }
  }

  const tail = parser.flush()
  if (tail.content && template) {
    out.push({
      type: "json",
      value: {
        ...template,
        choices: [{ index: 0, delta: { content: tail.content }, finish_reason: null }],
      },
    })
  }
  for (const call of tail.calls) {
    toolIndex++
    out.push({
      type: "json",
      value: {
        ...template,
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              tool_calls: [
                {
                  index: toolIndex - 1,
                  id: call.id,
                  type: "function",
                  function: { name: call.name, arguments: JSON.stringify(call.args) },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      },
    })
  }

  // If we synthesised any tool calls, make sure the final finish_reason reflects it.
  if (toolIndex > 0) {
    for (let i = out.length - 1; i >= 0; i--) {
      const ev = out[i]
      if (ev.type !== "json") continue
      const choice = ev.value?.choices?.[0]
      if (!choice) continue
      if (choice.finish_reason && choice.finish_reason !== "tool_calls") {
        choice.finish_reason = "tool_calls"
      }
      break
    }
  }

  return out
}

type ParsedSSEEvent =
  | { type: "done" }
  | { type: "raw"; data: string }
  | { type: "json"; value: Record<string, any> }

function transformLegacyChunkEvent(event: ParsedSSEEvent): ParsedSSEEvent {
  if (event.type !== "json") return event
  const value = JSON.parse(JSON.stringify(event.value))
  const choice = value?.choices?.[0]
  if (!choice) return event

  normalizeLegacyFunctionCall(choice)

  const delta = choice?.delta
  if (delta?.function_call && !delta.tool_calls) {
    const legacy = delta.function_call
    delta.tool_calls = [
      {
        index: 0,
        id: SYNTHETIC_TOOL_CALL_ID,
        type: "function",
        function: { name: legacy.name ?? undefined, arguments: legacy.arguments ?? "" },
      },
    ]
    delete delta.function_call
  }
  if (choice.finish_reason === "function_call") choice.finish_reason = "tool_calls"
  return { type: "json", value }
}

function normalizeLegacyFunctionCall(choice: Record<string, any>): void {
  if (choice?.message?.function_call && !choice?.message?.tool_calls) {
    const legacy = choice.message.function_call
    choice.message.tool_calls = [
      {
        id: SYNTHETIC_TOOL_CALL_ID,
        type: "function",
        function: { name: legacy.name, arguments: legacy.arguments ?? "" },
      },
    ]
    delete choice.message.function_call
  }
  if (choice?.finish_reason === "function_call") choice.finish_reason = "tool_calls"
}

function parseToolCallFromContent(
  content: unknown,
  parsers: OpenAICompatibleToolParser[],
): OpenAICompatibleToolCall | undefined {
  if (typeof content !== "string") return
  const trimmed = content.trim()
  if (!trimmed) return

  for (const parser of parsers) {
    if (parser.type === "json") {
      const parsed = parseJsonToolCall(trimmed)
      if (parsed) return parsed
    }
  }
}

function parseJsonToolCall(content: string): OpenAICompatibleToolCall | undefined {
  const candidates = [content, stripMarkdownCodeFence(content), extractTaggedContent(content, "tool_call")].filter(
    Boolean,
  ) as string[]

  for (const candidate of candidates) {
    const parsed = tryParseJson(candidate)
    const toolCall = parsed ? toToolCall(parsed) : undefined
    if (toolCall) return toolCall
  }
}

function toToolCall(parsed: any): OpenAICompatibleToolCall | undefined {
  const fromFunction = parsed?.function
  const name =
    asString(parsed?.name) ??
    asString(parsed?.tool) ??
    asString(parsed?.toolName) ??
    asString(fromFunction?.name) ??
    undefined
  if (!name) return

  const args = parsed?.arguments ?? parsed?.input ?? parsed?.args ?? fromFunction?.arguments
  const normalizedArguments = normalizeArguments(args)
  if (!normalizedArguments) return

  return {
    id: SYNTHETIC_TOOL_CALL_ID,
    type: "function",
    function: { name, arguments: normalizedArguments },
  }
}

function normalizeArguments(input: unknown): string | undefined {
  if (typeof input === "string") {
    if (tryParseJson(input)) return input
    return
  }
  if (input && typeof input === "object" && !Array.isArray(input)) {
    return JSON.stringify(input)
  }
}

function asString(input: unknown): string | undefined {
  return typeof input === "string" && input.trim() ? input.trim() : undefined
}

function stripMarkdownCodeFence(input: string): string | undefined {
  const match = input.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i)
  return match?.[1]?.trim()
}

function extractTaggedContent(input: string, tag: string): string | undefined {
  const match = input.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`, "i"))
  return match?.[1]?.trim()
}

function tryParseJson(input: string): any {
  try {
    return JSON.parse(input)
  } catch {
    return undefined
  }
}

function parseSSEEvents(text: string): ParsedSSEEvent[] {
  return text
    .split(/\n\n+/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block): ParsedSSEEvent => {
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n")

      if (data === "[DONE]") return { type: "done" }
      const parsed = tryParseJson(data)
      if (!parsed) return { type: "raw", data }
      return { type: "json", value: parsed }
    })
}

function serializeSSEEvents(events: ParsedSSEEvent[]): string {
  return events
    .map((event) => {
      if (event.type === "done") return "data: [DONE]\n\n"
      if (event.type === "raw") return `data: ${event.data}\n\n`
      return `data: ${JSON.stringify(event.value)}\n\n`
    })
    .join("")
}
