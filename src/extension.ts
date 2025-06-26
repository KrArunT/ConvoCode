import * as vscode from 'vscode';
import { Ollama } from 'ollama';

export function activate(context: vscode.ExtensionContext) {
  const config = vscode.workspace.getConfiguration('convocode');

  // Fallback to local Ollama API and a default model if none configured
  const baseUrl = config.get<string>('baseUrl') || 'http://127.0.0.1:11434';
  const defaultModel = config.get<string>('model') || 'qwen3:0.6b';
  const availableModels = config.get<string[]>('models') || [defaultModel];
  const ollama = new Ollama({ host: baseUrl });

  // Register the command that opens our chat panel
  const disposable = vscode.commands.registerCommand('convocode.openChat', () => {
    const panel = vscode.window.createWebviewPanel(
      'convocode',                // viewType identifier
      'ConvoCode Chat',           // panel title
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')],
      }
    );

    // Set up the initial HTML content, passing in model options
    panel.webview.html = getWebviewContent(panel.webview, defaultModel, availableModels);

    // Listen for messages from the webview
    panel.webview.onDidReceiveMessage(async (message) => {
      if (message.command === 'send') {
        const model = message.model || defaultModel;
        panel.webview.postMessage({ command: 'thinking' });
        try {
          const stream = await ollama.chat({
            model,
            messages: [{ role: 'user', content: message.text }],
            stream: true,
            think: true,
          });

          let started = false;
          for await (const chunk of stream) {
            if (chunk.message.thinking && !started) {
              panel.webview.postMessage({ command: 'updateThinking', text: chunk.message.thinking });
            }
            if (chunk.message.content) {
              if (!started) {
                started = true;
                panel.webview.postMessage({ command: 'startResponse' });
              }
              panel.webview.postMessage({ command: 'appendResponse', text: chunk.message.content });
            }
          }
        } catch (err) {
          panel.webview.postMessage({ command: 'error', text: (err as Error).message });
        }
      }
    });

    context.subscriptions.push(panel);
  });

  context.subscriptions.push(disposable);
}

function getWebviewContent(webview: vscode.Webview, defaultModel: string, models: string[]): string {
  const nonce = getNonce();
  const markedSrc = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';

  // Build <option> tags dynamically from availableModels
  const options = models
    .map(m => `<option value="${m}"${m === defaultModel ? ' selected' : ''}>${m}</option>`)
    .join('');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ConvoCode Chat</title>
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'nonce-${nonce}' https:; style-src 'unsafe-inline';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body { font-family: sans-serif; padding: 10px; margin: 0; }
    #chat { width: 100%; height: 70vh; border: 1px solid #ccc; padding: 8px; overflow-y: auto; resize: vertical; white-space: pre-wrap; }
    #controls { margin-top: 8px; display: flex; gap: 8px; }
    #model { width: 150px; }
    #input { flex: 1; min-height: 1.5em; resize: vertical; }
    .user-message { margin: 4px 0; }
    .assistant-message { margin: 4px 0; }
    .thinking { font-style: italic; color: grey; }
    .error { color: red; }
  </style>
  <script nonce="${nonce}" src="${markedSrc}"></script>
</head>
<body>
  <div id="chat"></div>
  <div id="controls">
    <select id="model">${options}</select>
    <textarea id="input" placeholder="Type your message..."></textarea>
    <button id="send">Send</button>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const chatEl = document.getElementById('chat');
    const inputEl = document.getElementById('input');
    const modelEl = document.getElementById('model');

    document.getElementById('send').addEventListener('click', () => {
      const text = inputEl.value.trim();
      if (!text) return;
      const model = modelEl.value;
      chatEl.insertAdjacentHTML('beforeend', '<div class="user-message"><strong>You:</strong> ' + marked.parse(text) + '</div>');
      vscode.postMessage({ command: 'send', text, model });
      inputEl.value = '';
      chatEl.scrollTop = chatEl.scrollHeight;
    });

    window.addEventListener('message', event => {
      const msg = event.data;
      switch (msg.command) {
        case 'thinking':
          chatEl.insertAdjacentHTML('beforeend', '<div class=\"thinking\">Thinking...</div>');
          break;
        case 'updateThinking':
          const thinkingEl = chatEl.querySelector('.thinking:last-of-type');
          if (thinkingEl) thinkingEl.textContent = msg.text;
          break;
        case 'startResponse':
          chatEl.insertAdjacentHTML('beforeend', '<div class=\"assistant-message\"><strong>Response:</strong> <span class=\"response-content\" data-raw=\"\"></span></div>');
          break;
        case 'appendResponse':
          const contentEl = chatEl.querySelector('.assistant-message:last-of-type .response-content');
          if (contentEl) {
            const raw = (contentEl.getAttribute('data-raw') || '') + msg.text;
            contentEl.setAttribute('data-raw', raw);
            contentEl.innerHTML = marked.parse(raw);
          }
          break;
        case 'error':
          chatEl.insertAdjacentHTML('beforeend', '<div class="assistant-message error"><strong>Error:</strong> ' + msg.text + '</div>');
          break;
      }
      chatEl.scrollTop = chatEl.scrollHeight;
    });
  </script>
</body>
</html>`;
}

function getNonce() {
  let text = '';
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 16; i++) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}

export function deactivate() {}
