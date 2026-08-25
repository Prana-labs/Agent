const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Send a question (and optional PDF file) to the backend chat endpoint.
 * If file is provided, it initiates a new session and returns a thread_id.
 * If threadId is provided, it continues the existing session with conversation history.
 *
 * @param {Object} params
 * @param {File} [params.file] - The PDF file to upload.
 * @param {string} params.question - The question to ask.
 * @param {string} [params.threadId] - The existing thread ID.
 * @returns {Promise<{thread_id: string, question: string, answer: string}>}
 */
export async function chatWithPdf({ file, files, question, threadId }) {
  const formData = new FormData();
  if (question !== undefined && question !== null) {
    formData.append('question', question);
  }
  
  if (files && files.length > 0) {
    files.forEach((f) => {
      formData.append('files', f);
    });
  } else if (file) {
    formData.append('files', file);
  }

  if (threadId) {
    formData.append('thread_id', threadId);
  }

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'An error occurred on the server.' }));
    throw new Error(errorData.detail || 'Failed to communicate with PDF RAG service.');
  }

  return response.json();
}
