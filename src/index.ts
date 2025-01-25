#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import fetch from 'node-fetch';
import { RequestInit } from 'node-fetch';

// Base configuration schema that all requests can include
const BaseConfigSchema = z.object({
    apiUrl: z.string().default('http://localhost:5001'),
});

// Core API schemas (api/v1)
const MaxContextLengthSchema = BaseConfigSchema;
const MaxLengthSchema = BaseConfigSchema;
const GenerateSchema = BaseConfigSchema.extend({
    prompt: z.string(),
    max_length: z.number().optional(),
    max_context_length: z.number().optional(),
    temperature: z.number().optional(),
    top_p: z.number().optional(),
    top_k: z.number().optional(),
    repetition_penalty: z.number().optional(),
    stop_sequence: z.array(z.string()).optional(),
    seed: z.number().optional(),
});

// Multiplayer schemas
const MultiplayerStatusSchema = BaseConfigSchema;
const MultiplayerGetStorySchema = BaseConfigSchema;
const MultiplayerSetStorySchema = BaseConfigSchema.extend({
    story: z.string()
});
// Store chat history in memory
const chatHistory: Array<{
    role: 'system' | 'user' | 'assistant',
    content: string
}> = [];
// Generate check schemas
const GenerateCheckSchema = BaseConfigSchema;
const GenerateCheckMultiuserSchema = BaseConfigSchema;

// Extra API schemas (api/extra)
const TokenCountSchema = BaseConfigSchema.extend({
    text: z.string(),
});

const DetokenizeSchema = BaseConfigSchema.extend({
    tokens: z.array(z.number()),
});

const TranscribeSchema = BaseConfigSchema.extend({
    audio: z.string(),
    language: z.string().optional(),
});

const WebSearchSchema = BaseConfigSchema.extend({
    query: z.string(),
});

const TTSSchema = BaseConfigSchema.extend({
    text: z.string(),
    voice: z.string().optional(),
    speed: z.number().optional(),
});

const AbortSchema = BaseConfigSchema;
const PerfInfoSchema = BaseConfigSchema;
const ModelInfoSchema = BaseConfigSchema;
const VersionInfoSchema = BaseConfigSchema;
const PreloadStorySchema = BaseConfigSchema;
const LastLogProbsSchema = BaseConfigSchema;

// Stable Diffusion API schemas (sdapi/v1)
const Txt2ImgSchema = BaseConfigSchema.extend({
    prompt: z.string(),
    negative_prompt: z.string().optional(),
    width: z.number().optional(),
    height: z.number().optional(),
    steps: z.number().optional(),
    cfg_scale: z.number().optional(),
    sampler_name: z.string().optional(),
    seed: z.number().optional(),
});

const Img2ImgSchema = Txt2ImgSchema.extend({
    init_images: z.array(z.string()),
    denoising_strength: z.number().optional(),
});

const InterrogateSchema = BaseConfigSchema.extend({
    image: z.string(),
});

const SDModelsSchema = BaseConfigSchema;
const SDSamplersSchema = BaseConfigSchema;
const SDOptionsSchema = BaseConfigSchema;

// OpenAI compatible API schemas (v1)
const ChatCompletionSchema = BaseConfigSchema.extend({
    messages: z.array(z.object({
        role: z.enum(['system', 'user', 'assistant']),
        content: z.string(),
    })),
    temperature: z.number().optional(),
    top_p: z.number().optional(),
    max_tokens: z.number().optional(),
    stop: z.array(z.string()).optional(),
});

const CompletionSchema = BaseConfigSchema.extend({
    prompt: z.string(),
    max_tokens: z.number().optional(),
    temperature: z.number().optional(),
    top_p: z.number().optional(),
    stop: z.array(z.string()).optional(),
});

const ModelsSchema = BaseConfigSchema;

const AudioTranscriptionSchema = BaseConfigSchema.extend({
    file: z.string(),
    model: z.string().optional(),
    language: z.string().optional(),
});

const AudioSpeechSchema = BaseConfigSchema.extend({
    input: z.string(),
    voice: z.string().optional(),
    speed: z.number().optional(),
});

// Server setup
const server = new Server({
    name: "kobold-server",
    version: "0.1.0",
}, {
    capabilities: {
        tools: {},
    },
});

// Helper function for HTTP requests
async function makeRequest(url: string, method = 'GET', body: Record<string, unknown> | null = null) {
    const options: RequestInit = {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
    };
    
    if (body && method !== 'GET') {
        options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);
    if (!response.ok) {
        throw new Error(`KoboldAI API error: ${response.statusText}`);
    }
    
    return response.json();
}

server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
        // Core API tools
        {
            name: "kobold_max_context_length",
            description: "Get current max context length setting",
            inputSchema: zodToJsonSchema(MaxContextLengthSchema),
        },
        {
            name: "kobold_max_length",
            description: "Get current max length setting",
            inputSchema: zodToJsonSchema(MaxLengthSchema),
        },
        {
            name: "kobold_generate",
            description: "Generate text with KoboldAI",
            inputSchema: zodToJsonSchema(GenerateSchema),
        },
        // Extra API tools
        {
            name: "kobold_model_info",
            description: "Get current model information",
            inputSchema: zodToJsonSchema(ModelInfoSchema),
        },
        {
            name: "kobold_version",
            description: "Get KoboldAI version information",
            inputSchema: zodToJsonSchema(VersionInfoSchema),
        },
        {
            name: "kobold_perf_info",
            description: "Get performance information",
            inputSchema: zodToJsonSchema(PerfInfoSchema),
        },
        {
            name: "kobold_token_count",
            description: "Count tokens in text",
            inputSchema: zodToJsonSchema(TokenCountSchema),
        },
        {
            name: "kobold_detokenize",
            description: "Convert token IDs to text",
            inputSchema: zodToJsonSchema(DetokenizeSchema),
        },
        {
            name: "kobold_transcribe",
            description: "Transcribe audio using Whisper",
            inputSchema: zodToJsonSchema(TranscribeSchema),
        },
        {
            name: "kobold_web_search",
            description: "Search the web via DuckDuckGo",
            inputSchema: zodToJsonSchema(WebSearchSchema),
        },
        {
            name: "kobold_tts",
            description: "Generate text-to-speech audio",
            inputSchema: zodToJsonSchema(TTSSchema),
        },
        {
            name: "kobold_abort",
            description: "Abort the currently ongoing generation",
            inputSchema: zodToJsonSchema(AbortSchema),
        },
        {
            name: "kobold_last_logprobs",
            description: "Get token logprobs from the last request",
            inputSchema: zodToJsonSchema(LastLogProbsSchema),
        },
        // Stable Diffusion API tools
        {
            name: "kobold_sd_models",
            description: "List available Stable Diffusion models",
            inputSchema: zodToJsonSchema(SDModelsSchema),
        },
        {
            name: "kobold_sd_samplers",
            description: "List available Stable Diffusion samplers",
            inputSchema: zodToJsonSchema(SDSamplersSchema),
        },
        {
            name: "kobold_txt2img",
            description: "Generate image from text prompt",
            inputSchema: zodToJsonSchema(Txt2ImgSchema),
        },
        {
            name: "kobold_img2img",
            description: "Transform existing image using prompt",
            inputSchema: zodToJsonSchema(Img2ImgSchema),
        },
        {
            name: "kobold_interrogate",
            description: "Generate caption for image",
            inputSchema: zodToJsonSchema(InterrogateSchema),
        },
        // OpenAI compatible API tools
        {
            name: "kobold_chat",
            description: "Chat completion (OpenAI-compatible)",
            inputSchema: zodToJsonSchema(ChatCompletionSchema),
        },
        {
            name: "kobold_complete",
            description: "Text completion (OpenAI-compatible)",
            inputSchema: zodToJsonSchema(CompletionSchema),
        },
    ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    try {
        const { name, arguments: args } = request.params;
        const { apiUrl = 'http://localhost:5001', ...requestData } = args as z.infer<typeof BaseConfigSchema>;
        // Special handling for chat completions
        if (name === 'kobold_chat') {
            // Add new messages to chat history
            const newMessages = (requestData as any).messages || [];
            chatHistory.push(...newMessages);

            // Get last 4 messages
            const recentMessages = chatHistory.slice(-4);
            console.error('Last 4 messages in chat:');
            console.error(JSON.stringify(recentMessages, null, 2));

            // Make the API request with all context
            const result = await makeRequest(
                `${apiUrl}/v1/chat/completions`,
                'POST',
                { ...requestData, messages: chatHistory }
            );

            // Add assistant's response to history
            const typedResult = result as any;
            if (typedResult.choices?.[0]?.message) {
                chatHistory.push(typedResult.choices[0].message);
            }

            return {
                content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
                isError: false,
            };
        }
        // Handle GET requests differently
        const getEndpoints: Record<string, string> = {
            kobold_max_context_length: '/api/v1/config/max_context_length',
            kobold_max_length: '/api/v1/config/max_length',
            kobold_generate_check: '/api/extra/generate/check',
            kobold_model_info: '/api/v1/model',
            kobold_version: '/api/v1/info/version',
            kobold_perf_info: '/api/extra/perf',
            kobold_sd_models: '/sdapi/v1/sd-models',
            kobold_sd_samplers: '/sdapi/v1/samplers',
        };

        if (getEndpoints[name]) {
            const result = await makeRequest(`${apiUrl}${getEndpoints[name]}`);
            return {
                content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
                isError: false,
            };
        }

        // Handle POST endpoints
        const postEndpoints: Record<string, { endpoint: string; schema: z.ZodTypeAny }> = {
            kobold_multiplayer_status: { endpoint: '/api/extra/multiplayer/status', schema: MultiplayerStatusSchema },
            kobold_multiplayer_get_story: { endpoint: '/api/extra/multiplayer/getstory', schema: MultiplayerGetStorySchema },
            kobold_multiplayer_set_story: { endpoint: '/api/extra/multiplayer/setstory', schema: MultiplayerSetStorySchema },
            kobold_generate_check_multiuser: { endpoint: '/api/extra/generate/check', schema: GenerateCheckMultiuserSchema },
            kobold_generate: { endpoint: '/api/v1/generate', schema: GenerateSchema },
            kobold_token_count: { endpoint: '/api/extra/tokencount', schema: TokenCountSchema },
            kobold_detokenize: { endpoint: '/api/extra/detokenize', schema: DetokenizeSchema },
            kobold_transcribe: { endpoint: '/api/extra/transcribe', schema: TranscribeSchema },
            kobold_web_search: { endpoint: '/api/extra/websearch', schema: WebSearchSchema },
            kobold_tts: { endpoint: '/api/extra/tts', schema: TTSSchema },
            kobold_abort: { endpoint: '/api/extra/abort', schema: AbortSchema },
            kobold_last_logprobs: { endpoint: '/api/extra/last_logprobs', schema: LastLogProbsSchema },
            kobold_txt2img: { endpoint: '/sdapi/v1/txt2img', schema: Txt2ImgSchema },
            kobold_img2img: { endpoint: '/sdapi/v1/img2img', schema: Img2ImgSchema },
            kobold_interrogate: { endpoint: '/sdapi/v1/interrogate', schema: InterrogateSchema },
            kobold_complete: { endpoint: '/v1/completions', schema: CompletionSchema },
        };

        if (postEndpoints[name]) {
            const { endpoint, schema } = postEndpoints[name];
            const parsed = schema.safeParse(args);
            if (!parsed.success) {
                throw new Error(`Invalid arguments: ${parsed.error}`);
            }

            const result = await makeRequest(`${apiUrl}${endpoint}`, 'POST', requestData);
            return {
                content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
                isError: false,
            };
        }

        throw new Error(`Unknown tool: ${name}`);
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        return {
            content: [{ type: "text", text: `Error: ${errorMessage}` }],
            isError: true,
        };
    }
});

// Start server
async function runServer() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("KoboldAI MCP server running on stdio");
}

runServer().catch((error) => {
    console.error("Fatal error running server:", error);
    process.exit(1);
});