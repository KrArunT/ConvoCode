
import { Ollama } from 'ollama';

async function listModels() {
  // Initialize the client (adjust host if needed)
  const ollama = new Ollama({ host: 'http://127.0.0.1:11434' });

  try {
    // Call the list() API to fetch available models
    const response = await ollama.list();
    console.log('Available models:', response.models);
  } catch (err) {
    console.error('Failed to list models:', err);
  }
}

listModels();
