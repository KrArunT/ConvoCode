// src/extension.ts
import * as vscode from 'vscode';
import { Ollama } from 'ollama';

export function activate(context: vscode.ExtensionContext) {
  const ollama = new Ollama({ host: 'http://127.0.0.1:11434' });

  const output = vscode.window.createOutputChannel('Ollama Chat');

  let disposable = vscode.commands.registerCommand('ollamaChat.start', async () => {
    while (true) {
      const userInput = await vscode.window.showInputBox({
        prompt: 'You',
        ignoreFocusOut: true
      });
      if (userInput === undefined) {
        break; // user cancelled
      }

      output.clear();
      output.appendLine('Thinking...');
      output.show();

      const responseStream = await ollama.chat({
        model: 'qwen3:0.6b',
        messages: [{ role: 'user', content: userInput }],
        stream: true,
        think: true
      });

      let started = false;
      for await (const chunk of responseStream) {
        if (chunk.message.thinking && !started) {
          output.append(chunk.message.thinking);
        }
        if (chunk.message.content) {
          if (!started) {
            started = true;
            output.appendLine('');
            output.appendLine('Response:');
            output.appendLine('========');
          }
          output.append(chunk.message.content);
        }
      }
    }
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {}
