import * as vscode from 'vscode';
import { Ollama } from 'ollama';

export function activate(context: vscode.ExtensionContext) {
  const config = vscode.workspace.getConfiguration('ollamaChat');
  const baseUrl = config.get<string>('baseUrl')!;
  const defaultModel = config.get<string>('model')!;
  const ollama = new Ollama({ host: baseUrl });

  const disposable = vscode.commands.registerCommand('ollamaChat.openChat', () => {
    const panel = vscode.window.createWebviewPanel(
      'ollamaChat',
      'Ollama Chat',
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')],
      }
    );

    panel.webview.html = getWebviewContent(panel.webview, defaultModel);

    panel.webview.onDidReceiveMessage(async message => {
      if (message.command === 'send') {
        const model = message.model || defaultModel;
        panel.webview.postMessage({ command: 'thinking' });
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
      }
    });

    context.subscriptions.push(panel);
  });

  context.subscriptions.push(disposable);
}

function getWebviewContent(webview: vscode.Webview, defaultModel: string): string {
  const nonce = getNonce();
  const markedSrc = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'nonce-${nonce}' https:; style-src 'unsafe-inline';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body { font-family: sans-serif; padding: 10px; margin: 0; }
    #chat {
      width: 100%;
      height: 70vh;
      border: 1px solid #ccc;
      padding: 8px;
      overflow-y: auto;
      resize: vertical;
      white-space: pre-wrap;
    }
    #controls { margin-top: 8px; display: flex; gap: 8px; }
    #model { width: 120px; }
    #input {
      flex: 1;
      min-height: 1.5em;
      resize: vertical;
    }
    .user-message { margin: 4px 0; }
    .assistant-message { margin: 4px 0; }
    .thinking { font-style: italic; color: grey; }
  </style>
  <script nonce="${nonce}" src="${markedSrc}"></script>
</head>
<body>
  <div id="chat"></div>
  <div id="controls">
    <select id="model">
      <option selected>${defaultModel}</option>
    </select>
    <textarea id="input" placeholder="Type your message..."></textarea>
    <button id="send">Send</button>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const chatEl = document.getElementById('chat');
    const inputEl = document.getElementById('input');
    document.getElementById('send').addEventListener('click', () => {
      const text = inputEl.value.trim();
      const model = document.getElementById('model').value;
      if (!text) return;
      // Render user input as Markdown blocks
      const userHtml = '<div class="user-message"><strong>You:</strong> ' + marked.parse(text) + '</div>';
      chatEl.insertAdjacentHTML('beforeend', userHtml);
      vscode.postMessage({ command: 'send', text, model });
      inputEl.value = '';
      chatEl.scrollTop = chatEl.scrollHeight;
    });

    window.addEventListener('message', event => {
      const msg = event.data;
      switch (msg.command) {
        case 'thinking':
          chatEl.insertAdjacentHTML('beforeend', '<div class="thinking">Thinking...</div>');
          break;
        case 'updateThinking': {
          const thinkingEl = chatEl.querySelector('.thinking:last-of-type');
          if (thinkingEl) thinkingEl.textContent = msg.text;
          break;
        }
        case 'startResponse':
          chatEl.insertAdjacentHTML('beforeend', '<div class="assistant-message"><strong>Response:</strong> <span class="response-content" data-raw=""></span></div>');
          break;
        case 'appendResponse': {
          const contentEl = chatEl.querySelector('.assistant-message:last-of-type .response-content');
          if (contentEl) {
            const raw = (contentEl.getAttribute('data-raw') || '') + msg.text;
            contentEl.setAttribute('data-raw', raw);
            // Render the combined raw Markdown as HTML
            contentEl.innerHTML = marked.parse(raw);
          }
          break;
        }
      }
      chatEl.scrollTop = chatEl.scrollHeight;
    });
  </script>
</body>
</html>`;
}

function getNonce() {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 16; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}

export function deactivate() {}
