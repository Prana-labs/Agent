import React, { useState, useRef, useEffect } from 'react';
import { chatWithPdf } from '../services/api';
import './comp.css';

export default function PdfChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [threadId, setThreadId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Restore session from localStorage if present
  useEffect(() => {
    const savedThreadId = localStorage.getItem('pdf_chat_thread_id');
    const savedFileName = localStorage.getItem('pdf_chat_file_name');
    const savedMessages = localStorage.getItem('pdf_chat_messages');
    
    if (savedThreadId && savedFileName) {
      setThreadId(savedThreadId);
      setFileName(savedFileName);
      if (savedMessages) {
        setMessages(JSON.parse(savedMessages));
      }
    }
  }, []);

  // Sync messages and session info to localStorage
  useEffect(() => {
    if (threadId && fileName) {
      localStorage.setItem('pdf_chat_thread_id', threadId);
      localStorage.setItem('pdf_chat_file_name', fileName);
      localStorage.setItem('pdf_chat_messages', JSON.stringify(messages));
    } else {
      localStorage.removeItem('pdf_chat_thread_id');
      localStorage.removeItem('pdf_chat_file_name');
      localStorage.removeItem('pdf_chat_messages');
    }
  }, [messages, threadId, fileName]);

  // Auto scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
        setFileName(droppedFile.name);
        setError(null);
      } else {
        setError('Please select a valid PDF file.');
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile);
        setFileName(selectedFile.name);
        setError(null);
      } else {
        setError('Please select a valid PDF file.');
      }
    }
  };

  const handleLoadPdf = async (e) => {
    e.stopPropagation();
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const result = await chatWithPdf({
        file: file,
        question: "Please briefly summarize this document and tell me what you can help with.",
        threadId: null
      });

      if (result.thread_id) {
        setThreadId(result.thread_id);
      }

      setMessages([
        { sender: 'ai', text: result.answer }
      ]);
    } catch (err) {
      setError(err.message || 'Failed to load PDF. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setFileName('');
    setThreadId('');
    setMessages([]);
    setError(null);
    setInput('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    if (!threadId && !file) {
      setError('Please upload a PDF file to start chatting.');
      return;
    }

    const userQuestion = input.trim();
    setInput('');
    setError(null);

    // Add user message to log
    const updatedMessages = [...messages, { sender: 'user', text: userQuestion }];
    setMessages(updatedMessages);
    setLoading(true);

    try {
      // Call stateful API
      const result = await chatWithPdf({
        file: threadId ? null : file, // Send file only on the first turn
        question: userQuestion,
        threadId: threadId || null
      });

      // Update state with result
      if (!threadId && result.thread_id) {
        setThreadId(result.thread_id);
      }

      setMessages((prev) => [...prev, { sender: 'ai', text: result.answer }]);
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
      // Remove the last user message since it failed to get a response
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Sidebar Section */}
      <aside className="sidebar">
        <div className="brand-section">
          <div className="brand-logo">P</div>
          <h2 className="brand-title">PDF RAG Chat</h2>
        </div>

        <div className="upload-section">
          <h3 className="upload-title">Document</h3>

          {!threadId ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', width: '100%' }}>
              <div 
                className={`dropzone ${dragActive ? 'active' : ''}`}
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  className="file-input"
                  accept=".pdf"
                  onChange={handleFileChange}
                />
                <svg className="dropzone-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" width="24" height="24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                {fileName ? (
                  <span className="dropzone-text file-name">{fileName}</span>
                ) : (
                  <>
                    <span className="dropzone-text">Click or drag PDF here</span>
                    <span className="dropzone-subtext">Supports PDF up to 25MB</span>
                  </>
                )}
              </div>
              
              {file && (
                <button 
                  className="load-pdf-button"
                  onClick={handleLoadPdf}
                  disabled={loading}
                >
                  {loading ? 'Ingesting PDF...' : 'Load PDF'}
                </button>
              )}
            </div>
          ) : (
            <div className="session-info">
              <span className="session-header">Active Session</span>
              <div className="file-details">
                <svg className="file-details-icon" fill="currentColor" viewBox="0 0 20 20" width="18" height="18">
                  <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                </svg>
                <span className="file-name" title={fileName}>{fileName}</span>
              </div>
              <span className="session-header">Thread ID</span>
              <span className="thread-badge" title={threadId}>{threadId}</span>
              
              <button className="reset-button" onClick={handleReset}>
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" width="16" height="16">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Reset Chat
              </button>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          v1.0.0 • Connected
        </div>
      </aside>

      {/* Main Chat Panel */}
      <main className="chat-area">
        <header className="chat-header">
          <div className="header-status">
            <span className="status-indicator"></span>
            <span className="status-text">{threadId ? 'Session Active' : 'Waiting for Upload'}</span>
          </div>
        </header>

        {/* Message Log */}
        <div className="messages-log">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">💬</div>
              <h2 className="welcome-title">Ask your PDF Anything</h2>
              <p className="welcome-desc">
                {!threadId 
                  ? 'First select or drag a PDF document in the sidebar, type your opening question below, and press send to begin.'
                  : 'Start asking questions about your document. The conversation is stateful and will remember your previous questions!'}
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`message-bubble ${msg.sender}`}>
                <div className="avatar">
                  {msg.sender === 'user' ? 'U' : 'AI'}
                </div>
                <div className="message-content">
                  {msg.text}
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="message-bubble ai">
              <div className="avatar">AI</div>
              <div className="typing-bubble">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Panel */}
        <div className="input-panel">
          {error && (
            <div className="error-bar">
              <span>{error}</span>
              <button className="error-close" onClick={() => setError(null)}>×</button>
            </div>
          )}

          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={!threadId && !file ? "Please upload a PDF to begin..." : "Type your question..."}
              disabled={loading}
            />
            <button 
              type="submit" 
              className="send-button"
              disabled={loading || !input.trim() || (!threadId && !file)}
            >
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" width="20" height="20">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
