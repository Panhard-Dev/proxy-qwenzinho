import test from "node:test";
import assert from "node:assert/strict";
import { StreamingToolParser } from "../tools/parser.ts";

const tools = [
  {
    type: "function" as const,
    function: {
      name: "Bash",
      description: "Run a shell command",
      parameters: {
        type: "object",
        properties: { command: { type: "string" } },
        required: ["command"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "read_file",
      description: "Read file",
      parameters: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "edit_file",
      description: "Edit file",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
          old_text: { type: "string" },
          new_text: { type: "string" },
        },
        required: ["path"],
      },
    },
  },
];

function collect(parser: StreamingToolParser, chunks: string[]) {
  const toolCalls: any[] = [];
  let text = "";
  for (const chunk of chunks) {
    const r = parser.feed(chunk);
    toolCalls.push(...r.toolCalls);
    text += r.text;
  }
  const f = parser.flush();
  toolCalls.push(...f.toolCalls);
  text += f.text;
  return { toolCalls, text };
}

test("token dialect: single <|tool_call_begin|> call parses", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls, text } = collect(parser, [
    '<|tool_call_begin|>Bash<|tool_call_argument_begin|>{"command": "grep playwright package.json"}<|tool_call_end|>',
  ]);
  assert.equal(toolCalls.length, 1);
  assert.equal(toolCalls[0].name, "Bash");
  assert.equal(toolCalls[0].arguments.command, "grep playwright package.json");
  assert.equal(text.trim(), "", "no marker text may leak");
});

test("token dialect: mutated markers without pipes / with slash", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls, text } = collect(parser, [
    '<tool_call_lines_begin|><tool_call_begin|>Bash<|tool_call_argument_begin|>{"command": "ls"} </tool_call_end|>',
  ]);
  assert.equal(toolCalls.length, 1);
  assert.equal(toolCalls[0].name, "Bash");
  assert.equal(toolCalls[0].arguments.command, "ls");
  assert.equal(text.trim(), "", "no marker text may leak");
});

test("token dialect: multiple consecutive calls", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls, text } = collect(parser, [
    '<|tool_call_begin|>read_file<|tool_call_argument_begin|>{"path": "a.ts"}<|tool_call_end|>' +
      '<|tool_call_begin|>read_file<|tool_call_argument_begin|>{"path": "b.ts"}<|tool_call_end|>',
  ]);
  assert.equal(toolCalls.length, 2);
  assert.equal(toolCalls[0].arguments.path, "a.ts");
  assert.equal(toolCalls[1].arguments.path, "b.ts");
  assert.equal(text.trim(), "");
});

test("token dialect: markers split across feed() chunks", () => {
  const parser = new StreamingToolParser(tools);
  const full =
    'Vou ler o arquivo.\n<|tool_call_begin|>read_file<|tool_call_argument_begin|>{"path": "src/index.ts"}<|tool_call_end|>';
  // split in awkward places, including mid-marker
  const { toolCalls, text } = collect(parser, [
    "Vou ler o arquivo.\n<|tool_ca",
    "ll_begin|>read_file<|tool_call_arg",
    'ument_begin|>{"path": "src/in',
    'dex.ts"}<|tool_call_e',
    "nd|>",
  ]);
  assert.equal(toolCalls.length, 1, "call must parse across chunk splits");
  assert.equal(toolCalls[0].name, "read_file");
  assert.equal(toolCalls[0].arguments.path, "src/index.ts");
  assert.ok(!text.includes("<|"), "no marker fragments may leak as text");
});

test("token dialect: DeepSeek functions.NAME:IDX name form", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls } = collect(parser, [
    '<|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>{"path": "x.ts"}<|tool_call_end|>',
  ]);
  assert.equal(toolCalls.length, 1);
  assert.equal(toolCalls[0].name, "read_file");
});

test("token dialect: undeclared name tracked as malformed, never leaks", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls, text } = collect(parser, [
    '<|tool_call_begin|>launch_missiles<|tool_call_argument_begin|>{"target": "moon"}<|tool_call_end|>',
  ]);
  assert.equal(toolCalls.length, 0);
  assert.ok(!text.includes("tool_call_begin"), "markers must not leak");
  const malformed = parser.getMalformedToolCalls();
  assert.equal(malformed.length, 1);
  assert.deepEqual(malformed[0].undeclaredNames, ["launch_missiles"]);
});

test("token dialect: truncated arguments at stream end tracked as malformed", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls, text } = collect(parser, [
    '<|tool_call_begin|>edit_file<|tool_call_argument_begin|>{"path": "a.ts", "old_text": "incomple',
  ]);
  assert.equal(toolCalls.length, 0, "truncated call must not be fabricated");
  assert.ok(!text.includes("tool_call"), "markers must not leak");
  assert.equal(parser.getMalformedToolCalls().length, 1);
});

test("token dialect: prose before call is held as lead-in (not emitted with tool call)", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls, text } = collect(parser, [
    'Deixa eu verificar. <|tool_call_begin|>Bash<|tool_call_argument_begin|>{"command": "ls"}<|tool_call_end|>',
  ]);
  assert.equal(toolCalls.length, 1);
  assert.equal(text.trim(), "", "lead-in is suppressed once a tool call is emitted");
});

test("suffixed close tag </tool_call_read> closes a valid payload and never leaks", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls, text } = collect(parser, [
    '<tool_call>{"name": "read_file", "arguments": {"path": "a.ts"}}</tool_call_read>',
  ]);
  assert.equal(toolCalls.length, 1);
  assert.equal(toolCalls[0].name, "read_file");
  assert.ok(!text.includes("tool_call"), "mutated close tag must not leak");
});

test("suffixed close tag </tool_call_edit> in flush-recovery path", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls, text } = collect(parser, [
    '<tool_call>\n{"name": "edit_file", "arguments": {"path": "a.ts", "old_text": "x", "new_text": "y"}}\n</tool_call_edit>',
  ]);
  assert.equal(toolCalls.length, 1);
  assert.equal(toolCalls[0].name, "edit_file");
  assert.ok(!text.includes("tool_call"));
});

test("literal token markers inside inline code are preserved as text", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls, text } = collect(parser, [
    "O formato é `<|tool_call_begin|>` seguido do nome.",
  ]);
  assert.equal(toolCalls.length, 0);
  assert.ok(text.includes("<|tool_call_begin|>"), "quoted marker is prose");
});

test("mixed: text + token call + xml call in one stream", () => {
  const parser = new StreamingToolParser(tools);
  const { toolCalls } = collect(parser, [
    '<|tool_call_begin|>read_file<|tool_call_argument_begin|>{"path": "a.ts"}<|tool_call_end|>\n' +
      '<tool_call>{"name": "read_file", "arguments": {"path": "b.ts"}}</tool_call>',
  ]);
  assert.equal(toolCalls.length, 2);
  assert.equal(toolCalls[0].arguments.path, "a.ts");
  assert.equal(toolCalls[1].arguments.path, "b.ts");
});
