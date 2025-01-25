# Kobold MCP Server

A Model Context Protocol (MCP) server implementation for interfacing with KoboldAI. This server enables integration between KoboldAI's text generation capabilities and MCP-compatible applications.

## Features

- Text generation with KoboldAI
- Chat completion with persistent memory
- OpenAI-compatible API endpoints
- Stable Diffusion integration
- Built on the official MCP SDK
- TypeScript implementation

<a href="https://glama.ai/mcp/servers/a2xd4hoij7"><img width="380" height="200" src="https://glama.ai/mcp/servers/a2xd4hoij7/badge" alt="Kobold Server MCP server" /></a>

## Installation

```bash
npm install kobold-mcp-server
```

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn package manager
- Running KoboldAI instance

## Usage

```typescript
import { KoboldMCPServer } from 'kobold-mcp-server';

// Initialize the server
const server = new KoboldMCPServer();

// Start the server
server.start();
```

## Configuration

The server can be configured through environment variables or a configuration object:

```typescript
const config = {
  apiUrl: 'http://localhost:5001' // KoboldAI API endpoint
};
const server = new KoboldMCPServer(config);
```

## Supported APIs

- Core KoboldAI API (text generation, model info)
- Chat completion with conversation memory
- Text completion (OpenAI-compatible)
- Stable Diffusion integration (txt2img, img2img)
- Audio transcription and text-to-speech
- Web search capabilities

## Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kobold-mcp-server.git
cd kobold-mcp-server
```

2. Install dependencies:
```bash
npm install
```

3. Build the project:
```bash
npm run build
```

## Dependencies

- `@modelcontextprotocol/sdk`: ^1.0.1
- `node-fetch`: ^2.6.1
- `zod`: ^3.20.0
- `zod-to-json-schema`: ^3.23.5

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Support

For issues and feature requests, please use the GitHub issue tracker.
