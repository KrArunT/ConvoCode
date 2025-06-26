import { Ollama } from 'ollama';

async function listModelNames() {
  // Initialize the client
  const ollama = new Ollama({ host: 'http://127.0.0.1:11434' });

  try {
    // Fetch the list of models
    const { models } = await ollama.list();

    // Extract only the name property into a new array
    const modelNames = models.map(model => model.name);

    // Print out the names
    console.log('Available model names:', modelNames);
    return modelNames;
  } catch (err) {
    console.error('Error listing models:', err);
    return [];
  }
}

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
listModelNames();
