# ConvoCode Chat

A Visual Studio Code extension that lets you chat with Ollama models directly from your editor.

## Features

* Send messages to any local or remote Ollama model
* Live streaming of responses with partial updates
* Markdown rendering in chat history
* Select from one or more configured models
* Easy build, debug, and package workflow

## Prerequisites

* Install Ollama
* `curl -fsSL https://ollama.com/install.sh | sh`
* `ollama run qwen3:0.6b`
* [Node.js](https://nodejs.org/) v14 or higher
* [Visual Studio Code](https://code.visualstudio.com/) v1.70.0 or higher
* A running Ollama API server (by default: `http://127.0.0.1:11434`)

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourname/convocode.git
cd convocode
npm install
```

## Development

1. Compile TypeScript and watch for changes:

   ```bash
   npm run watch
   ```

2. Launch the Extension Development Host:

   * Open this folder in VS Code
   * Press `F5` to launch a new window with ConvoCode loaded

3. In the development host, open the command palette (`Ctrl+Shift+P`) and run:

   ```
   ConvoCode Chat: Start Conversation
   ```

4. Type your messages into the input box and hit **Send**.

## Configuration

Settings can be modified in your VS Code `settings.json` under the `convocode` namespace:

```json
{
  "convocode.baseUrl": "http://127.0.0.1:11434",       // Ollama API endpoint
  "convocode.model": "qwen3:0.6b",                     // Default model
  "convocode.models": ["qwen3:0.6b", "other-model:tag"]  // Available models
}
```

## Packaging & Publishing

1. Compile the extension:

   ```bash
   npm run compile
   ```

2. Create a VSIX package:

   ```bash
   npx vsce package
   ```

3. Install the generated `.vsix` locally:

   ```bash
   code --install-extension convocode-0.0.1.vsix
   ```

4. (Optional) Publish to the VS Code Marketplace:

   ```bash
   npx vsce publish
   ```

## Release Notes

### 0.0.1

* Initial release with streaming chat, model selection, and markdown support.

---

> *Contributions, issues, and feature requests are welcome!*
> *Developed  by Your Arun Kumar Tiwary*
