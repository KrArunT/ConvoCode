#!/usr/bin/env node

import { Ollama } from 'ollama';
import readline from 'readline';
import chalk from 'chalk';
import boxen from 'boxen';

// Initialize Ollama client
const ollama = new Ollama({ host: 'http://127.0.0.1:11434' });

async function chat() {
  // Setup readline interface with a styled prompt
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: chalk.blueBright('You: '),
  });

  rl.prompt();

  rl.on('line', async (line) => {
    const userMessage = line.trim();
    if (!userMessage) {
      rl.prompt();
      return;
    }

    // Show thinking box
    console.log(
      boxen(chalk.yellow('Thinking...'), {
        padding: 1,
        borderColor: 'yellow',
        align: 'center',
      })
    );

    // Stream the AI response
    const response = await ollama.chat({
      model: 'qwen3:0.6b',
      messages: [{ role: 'user', content: userMessage }],
      stream: true,
      think: true,
    });

    let started = false;
    for await (const chunk of response) {
      if (chunk.message.thinking && !started) {
        // Print thinking stream
        process.stdout.write(chalk.gray(chunk.message.thinking));
      }
      if (chunk.message.content) {
        if (!started) {
          started = true;
          // Separator before actual response
          console.log(chalk.green('\nResponse:\n========'));
        }
        process.stdout.write(chalk.white(chunk.message.content));
      }
    }

    console.log('\n');
    rl.prompt();
  });

  rl.on('close', () => {
    console.log(chalk.blue('Goodbye!'));
    process.exit(0);
  });
}

chat();
