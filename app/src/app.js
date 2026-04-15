// ── AURA App v2.0 — model + memory + web search + vision + agent mode ──
const OLLAMA_URL = 'http://localhost:11434';
const MODEL = 'aura';
const MEMORY_URL = 'http://localhost:23100';
const SEARCH_PROXY_URL = 'http://localhost:11435';

let conversations = JSON.parse(localStorage.getItem('aura_convs') || '[]');
let currentConvId = null;
let isStreaming = false;
let agentMode = false;
let pendingImages = []; // {base64, name}

const $ = id => document.getElementById(id);

// ── Init ──
window.addEventListener('DOMContentLoaded', () => {
  checkOllama();
  ensureMemoryRunning();
  renderConversations();
  bindEvents();
  setInterval(checkOllama, 10000);
  initTauriEvents();
});

// ── Ollama health check ──
async function checkOllama() {
  try {
    const r = await fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      $('statusDot').className = 'status-dot online';
      $('statusText').textContent = 'AURA Ready';
    } else throw new Error();
  } catch {
    $('statusDot').className = 'status-dot offline';
    $('statusText').textContent = 'Starting...';
  }
}

// ── MemoryPilot ──
async function ensureMemoryRunning() {
  try {
    const r = await fetch(`${MEMORY_URL}/health`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) return;
  } catch {}
  if (window.__TAURI__) {
    try { await window.__TAURI__.core.invoke('start_memorypilot'); } catch {}
  }
}

async function recallMemories(query) {
  try {
    const r = await fetch(`${MEMORY_URL}/search`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit: 5 }), signal: AbortSignal.timeout(3000),
    });
    const data = await r.json();
    if (data.results?.length > 0) return '\n\n[Relevant memories]\n' + data.results.map(m => m.content).join('\n');
  } catch {}
  return '';
}

async function saveToMemory(userMsg, assistantMsg) {
  try {
    await fetch(`${MEMORY_URL}/add`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: `User: ${userMsg}\nAURA: ${assistantMsg.slice(0, 500)}`, kind: 'conversation', tags: ['aura', 'chat'] }),
    });
  } catch {}
}

// ── Web Search ──
async function webSearch(query) {
  try {
    const r = await fetch(`${SEARCH_PROXY_URL}/search`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, num_results: 5 }), signal: AbortSignal.timeout(10000),
    });
    const data = await r.json();
    if (data.results) return data.results.map(r => `${r.title}: ${r.snippet} (${r.url})`).join('\n\n');
  } catch {}
  try {
    const r = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`, { signal: AbortSignal.timeout(8000) });
    const html = await r.text();
    const results = [];
    const matches = html.matchAll(/<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)<\/a>[\s\S]*?<a class="result__snippet"[^>]*>([^<]+)<\/a>/g);
    for (const m of matches) { if (results.length >= 5) break; results.push(`${m[2].trim()}: ${m[3].trim()} (${m[1]})`); }
    return results.join('\n\n') || 'No results found.';
  } catch { return 'Search unavailable.'; }
}

function needsWebSearch(text) {
  const triggers = [/search\s+(for|the|about)/i, /look\s+up/i, /find\s+(me|info|information|out)/i,
    /what('s| is)\s+(the\s+)?(latest|current|today|news)/i, /cherche/i, /recherche/i, /actualit/i,
    /prix\s+(de|du|actuel)/i, /who\s+(is|won|was)/i, /when\s+(is|did|was)/i, /how\s+much/i];
  return triggers.some(r => r.test(text));
}

// ── Image handling ──
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function handleImageUpload(files) {
  for (const file of files) {
    if (file.type.startsWith('image/')) {
      const base64 = await fileToBase64(file);
      pendingImages.push({ base64, name: file.name, type: file.type, kind: 'image' });
    } else if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
      const base64 = await fileToBase64(file);
      pendingImages.push({ base64, name: file.name, type: 'application/pdf', kind: 'pdf' });
    }
  }
  renderImagePreviews();
}

function renderImagePreviews() {
  const preview = $('imagePreview');
  if (pendingImages.length === 0) { preview.classList.add('hidden'); preview.innerHTML = ''; return; }
  preview.classList.remove('hidden');
  preview.innerHTML = pendingImages.map((img, i) => {
    if (img.kind === 'pdf') {
      return `<div class="image-preview-item">
        <div class="pdf-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="4" y="2" width="16" height="20" rx="2" stroke="currentColor" stroke-width="1.5"/><text x="12" y="15" text-anchor="middle" fill="currentColor" font-size="7" font-weight="bold">PDF</text></svg>
          ${img.name.slice(0, 12)}
        </div>
        <button class="remove-img" data-idx="${i}">×</button>
      </div>`;
    }
    return `<div class="image-preview-item">
      <img src="data:${img.type};base64,${img.base64}" alt="${img.name}">
      <button class="remove-img" data-idx="${i}">×</button>
    </div>`;
  }).join('');
  preview.querySelectorAll('.remove-img').forEach(btn => {
    btn.addEventListener('click', () => { pendingImages.splice(+btn.dataset.idx, 1); renderImagePreviews(); });
  });
  $('sendBtn').disabled = false;
}

// ── Agent Mode: tool definitions for Gemma 4 tool calling ──
const AGENT_TOOLS = [
  {
    type: 'function',
    function: {
      name: 'execute_command',
      description: 'Execute a shell command on the local macOS computer and return the output. Use for: running scripts, installing packages, file operations, system tasks, git commands, etc.',
      parameters: {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'The bash/zsh command to execute' }
        },
        required: ['command']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read the contents of a file on the local computer.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Absolute path to the file to read' }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Write content to a file on the local computer. Creates the file if it does not exist.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Absolute path to the file' },
          content: { type: 'string', description: 'Content to write' }
        },
        required: ['path', 'content']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'list_directory',
      description: 'List files and directories at the given path.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Absolute path to the directory' }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'web_search',
      description: 'Search the internet for current information.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Search query' }
        },
        required: ['query']
      }
    }
  }
];

// ── Execute tool calls from agent mode ──
async function executeTool(name, args) {
  try {
    switch (name) {
      case 'execute_command': {
        const resp = await fetch('http://localhost:11436/exec', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ command: args.command }),
          signal: AbortSignal.timeout(30000),
        });
        const data = await resp.json();
        return data.output || data.error || 'Command executed (no output)';
      }
      case 'read_file': {
        const resp = await fetch('http://localhost:11436/read', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: args.path }),
          signal: AbortSignal.timeout(10000),
        });
        const data = await resp.json();
        return data.content || data.error || 'File is empty';
      }
      case 'write_file': {
        const resp = await fetch('http://localhost:11436/write', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: args.path, content: args.content }),
          signal: AbortSignal.timeout(10000),
        });
        const data = await resp.json();
        return data.success ? `Written to ${args.path}` : (data.error || 'Write failed');
      }
      case 'list_directory': {
        const resp = await fetch('http://localhost:11436/ls', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: args.path }),
          signal: AbortSignal.timeout(10000),
        });
        const data = await resp.json();
        return data.entries ? data.entries.join('\n') : (data.error || 'Empty directory');
      }
      case 'web_search': {
        return await webSearch(args.query);
      }
      default: return `Unknown tool: ${name}`;
    }
  } catch (err) { return `Tool error: ${err.message}`; }
}

// ── Conversations ──
function newConversation() {
  const conv = { id: Date.now().toString(), title: 'New chat', messages: [] };
  conversations.unshift(conv);
  currentConvId = conv.id;
  saveConversations();
  renderConversations();
  $('chatMessages').innerHTML = '';
  $('chatMessages').classList.remove('hidden');
  $('welcomeScreen').classList.add('hidden');
  $('messageInput').focus();
}

function saveConversations() { localStorage.setItem('aura_convs', JSON.stringify(conversations)); }

function renderConversations() {
  const list = $('conversationsList');
  list.innerHTML = conversations.map(c => `
    <div class="conv-item ${c.id === currentConvId ? 'active' : ''}" data-id="${c.id}">
      <span class="conv-title">${c.title}</span>
      <button class="conv-delete" data-id="${c.id}">×</button>
    </div>
  `).join('');
  list.querySelectorAll('.conv-item').forEach(el => {
    el.addEventListener('click', () => loadConversation(el.dataset.id));
  });
  list.querySelectorAll('.conv-delete').forEach(el => {
    el.addEventListener('click', e => { e.stopPropagation(); deleteConversation(el.dataset.id); });
  });
}

function loadConversation(id) {
  currentConvId = id;
  const conv = conversations.find(c => c.id === id);
  if (!conv) return;
  $('chatMessages').classList.remove('hidden');
  $('welcomeScreen').classList.add('hidden');
  $('chatMessages').innerHTML = conv.messages.map(m => messageHTML(m.role, m.content, m.images)).join('');
  $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
  renderConversations();
}

function deleteConversation(id) {
  conversations = conversations.filter(c => c.id !== id);
  if (currentConvId === id) { currentConvId = null; $('chatMessages').classList.add('hidden'); $('welcomeScreen').classList.remove('hidden'); }
  saveConversations();
  renderConversations();
}

function messageHTML(role, content, images) {
  const cls = role === 'user' ? 'user-message' : 'assistant-message';
  const label = role === 'user' ? 'You' : 'AURA';
  const escaped = content.replace(/</g, '&lt;').replace(/>/g, '&gt;');
  let imagesHtml = '';
  if (images && images.length > 0) {
    imagesHtml = images.map(img => `<img class="msg-image" src="data:${img.type};base64,${img.base64}" alt="uploaded">`).join('');
  }
  return `<div class="message ${cls}"><div class="msg-role">${label}</div>${imagesHtml}<div class="msg-content">${escaped}</div></div>`;
}

function toolBlockHTML(label, content) {
  const escaped = content.replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return `<div class="tool-block"><div class="tool-label">${label}</div>${escaped}</div>`;
}

// ── Send message with memory + search + vision + agent ──
async function sendMessage() {
  const input = $('messageInput');
  const text = input.value.trim();
  if ((!text && pendingImages.length === 0) || isStreaming) return;

  if (!currentConvId) newConversation();
  const conv = conversations.find(c => c.id === currentConvId);

  // Build message with images and PDFs
  const msgData = { role: 'user', content: text || '(file)', images: [] };
  const sentImages = [...pendingImages];
  let pdfContext = '';
  if (sentImages.length > 0) {
    const imageFiles = sentImages.filter(i => i.kind === 'image');
    const pdfFiles = sentImages.filter(i => i.kind === 'pdf');
    msgData.images = imageFiles.map(i => ({ base64: i.base64, type: i.type }));
    // Extract text from PDFs
    for (const pdf of pdfFiles) {
      try {
        $('statusText').textContent = `Reading ${pdf.name}...`;
        const resp = await fetch('http://localhost:11436/extract-pdf', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ base64: pdf.base64 }),
          signal: AbortSignal.timeout(20000),
        });
        const data = await resp.json();
        pdfContext += `\n\n[PDF: ${pdf.name}]\n${data.text || data.error || 'Could not read PDF'}`;
      } catch { pdfContext += `\n\n[PDF: ${pdf.name} - extraction failed]`; }
    }
    msgData.pdfNames = pdfFiles.map(p => p.name);
    $('statusText').textContent = 'AURA Ready';
  }
  if (pdfContext) msgData.content = (text || 'Analyze this PDF') + pdfContext;
  conv.messages.push(msgData);
  $('chatMessages').innerHTML += messageHTML('user', text || '(image)', msgData.images);
  input.value = '';
  input.style.height = 'auto';
  $('sendBtn').disabled = true;
  pendingImages = [];
  renderImagePreviews();

  if (conv.messages.length === 1) {
    conv.title = (text || 'Image chat').slice(0, 40) + ((text || '').length > 40 ? '...' : '');
    renderConversations();
  }

  // Recall memories
  const memories = await recallMemories(text || 'image analysis');

  // Web search if needed
  let searchContext = '';
  if (text && needsWebSearch(text)) {
    $('statusText').textContent = 'Searching the web...';
    const results = await webSearch(text);
    searchContext = '\n\n[Web search results]\n' + results;
    $('statusText').textContent = 'AURA Ready';
  }

  const agentInstructions = agentMode ? `
You have access to tools to interact with the user's computer:
- execute_command: run any bash/zsh command
- read_file: read file contents
- write_file: create or write files
- list_directory: list files in a directory
- web_search: search the internet
Use tools when the user asks you to do something on their computer. You can chain multiple tool calls.` : '';

  const systemMsg = {
    role: 'system',
    content: `You are AURA (Autonomous Unified Reasoning Architecture). You are a powerful, private, local AI assistant running on the user's Mac.
You remember past conversations automatically.
When web search results are provided, use them to give accurate answers. Cite sources.
If images are sent, analyze them and respond about what you see.
Be concise, precise, and helpful. Respond in the user's language.${agentInstructions}${memories}${searchContext}`,
  };

  // Build Ollama messages with images (base64 for vision)
  const ollamaMessages = [systemMsg];
  for (const m of conv.messages) {
    const msg = { role: m.role, content: m.content };
    if (m.images && m.images.length > 0) {
      msg.images = m.images.map(i => i.base64);
    }
    ollamaMessages.push(msg);
  }

  // Stream response
  isStreaming = true;
  $('statusText').textContent = agentMode ? 'Agent thinking...' : 'Thinking...';
  $('sendBtn').disabled = true;
  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'message assistant-message';
  assistantDiv.innerHTML = `<div class="msg-role">AURA</div><div class="msg-content" id="streamTarget"><div class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div></div>`;
  $('chatMessages').appendChild(assistantDiv);
  $('chatMessages').scrollTop = $('chatMessages').scrollHeight;

  let fullResponse = '';
  try {
    const requestBody = {
      model: MODEL,
      messages: ollamaMessages,
      stream: true,
      keep_alive: '30m',
      options: { num_ctx: 16384 },
    };
    if (agentMode) {
      requestBody.tools = AGENT_TOOLS;
      requestBody.stream = false; // Tool calling requires non-streaming
    }

    const r = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    if (agentMode && !requestBody.stream) {
      // Non-streaming agent mode with tool calling loop
      let data = await r.json();
      const target = document.getElementById('streamTarget');
      const typingInd = target?.querySelector('.typing-indicator');
      if (typingInd) typingInd.remove();

      // Tool calling loop (max 10 iterations)
      let iterations = 0;
      while (data.message?.tool_calls && data.message.tool_calls.length > 0 && iterations < 10) {
        iterations++;
        // Show tool calls in UI
        for (const tc of data.message.tool_calls) {
          const toolName = tc.function.name;
          const toolArgs = tc.function.arguments;
          $('statusText').textContent = `Running: ${toolName}...`;

          // Show tool block in chat
          const toolDiv = document.createElement('div');
          toolDiv.innerHTML = toolBlockHTML(`Tool: ${toolName}`, JSON.stringify(toolArgs, null, 2));
          assistantDiv.querySelector('.msg-content').appendChild(toolDiv.firstChild);

          // Execute tool
          const toolResult = await executeTool(toolName, toolArgs);

          // Show result
          const resultDiv = document.createElement('div');
          resultDiv.innerHTML = toolBlockHTML('Result', toolResult.slice(0, 2000));
          assistantDiv.querySelector('.msg-content').appendChild(resultDiv.firstChild);

          // Add to conversation for context
          ollamaMessages.push(data.message);
          ollamaMessages.push({ role: 'tool', content: toolResult.slice(0, 4000) });
        }

        // Call model again with tool results
        $('statusText').textContent = 'Agent thinking...';
        const r2 = await fetch(`${OLLAMA_URL}/api/chat`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: MODEL, messages: ollamaMessages, tools: AGENT_TOOLS, stream: false, keep_alive: '30m', options: { num_ctx: 16384 } }),
        });
        data = await r2.json();
      }

      // Final text response
      fullResponse = data.message?.content || '';
      if (target) target.textContent = fullResponse;

    } else {
      // Streaming mode (normal chat)
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        for (const line of chunk.split('\n').filter(Boolean)) {
          try {
            const data = JSON.parse(line);
            if (data.message?.content) {
              fullResponse += data.message.content;
              const target = document.getElementById('streamTarget');
              if (target) {
                const typingInd = target.querySelector('.typing-indicator');
                if (typingInd) typingInd.remove();
                target.textContent = fullResponse;
              }
              $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
            }
          } catch {}
        }
      }
    }
  } catch (err) {
    fullResponse = 'Connection error. Is Ollama running?';
    const target = document.getElementById('streamTarget');
    if (target) {
      const typingInd = target.querySelector('.typing-indicator');
      if (typingInd) typingInd.remove();
      target.textContent = fullResponse;
    }
  }

  const target = document.getElementById('streamTarget');
  if (target) target.removeAttribute('id');

  conv.messages.push({ role: 'assistant', content: fullResponse });
  saveConversations();
  isStreaming = false;
  $('statusText').textContent = 'AURA Ready';
  $('sendBtn').disabled = !$('messageInput').value.trim();
  saveToMemory(text || 'image', fullResponse);
}

// ── Event bindings ──
function bindEvents() {
  $('sendBtn').addEventListener('click', sendMessage);
  $('newChatBtn').addEventListener('click', newConversation);

  // Image upload
  $('imageBtn').addEventListener('click', () => $('imageInput').click());
  $('imageInput').addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleImageUpload(e.target.files);
    e.target.value = '';
  });

  // Drag and drop anywhere in the window
  let dragCounter = 0;
  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    $('dropOverlay').classList.remove('hidden');
  });
  document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) { dragCounter = 0; $('dropOverlay').classList.add('hidden'); }
  });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    $('dropOverlay').classList.add('hidden');
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/') || f.type === 'application/pdf' || f.name.endsWith('.pdf'));
    if (files.length > 0) handleImageUpload(files);
  });

  // Paste images from clipboard
  document.addEventListener('paste', (e) => {
    const files = Array.from(e.clipboardData?.files || []).filter(f => f.type.startsWith('image/'));
    if (files.length > 0) { e.preventDefault(); handleImageUpload(files); }
  });

  // Agent mode toggle
  $('agentBtn').addEventListener('click', () => {
    agentMode = !agentMode;
    $('agentBtn').classList.toggle('active', agentMode);
    $('agentStatus').textContent = agentMode ? 'Agent mode ON' : '';
    $('messageInput').placeholder = agentMode ? 'Ask AURA to do something on your Mac...' : 'Ask AURA anything...';
  });

  // Text input
  $('messageInput').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    $('sendBtn').disabled = !this.value.trim() && pendingImages.length === 0;
  });
  $('messageInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
}

// ── Tauri init ──
async function initTauriEvents() {
  if (!window.__TAURI__) return;
  try {
    const { invoke } = window.__TAURI__.core;
    const r = await fetch(`${OLLAMA_URL}/api/tags`);
    const data = await r.json();
    const hasAura = data.models?.some(m => m.name.includes('aura'));
    if (!hasAura) {
      $('setupScreen').classList.remove('hidden');
      $('app').classList.add('hidden');
      await invoke('setup_aura_model');
      $('setupScreen').classList.add('hidden');
      $('app').classList.remove('hidden');
    }
  } catch {}
}

// ── Auto-update system ──
const CURRENT_VERSION = '0.1.0';
const UPDATE_CHECK_URL = 'https://api.github.com/repos/Soflutionltd/aura/releases/latest';

async function checkForUpdate() {
  try {
    const r = await fetch(UPDATE_CHECK_URL, {
      headers: { 'Accept': 'application/vnd.github.v3+json' },
      signal: AbortSignal.timeout(10000),
    });
    if (!r.ok) return;
    const release = await r.json();
    const latestVersion = release.tag_name?.replace('v', '') || '';
    if (latestVersion && latestVersion !== CURRENT_VERSION) {
      const dmgAsset = release.assets?.find(a => a.name.endsWith('.dmg'));
      showUpdatePopup({
        version: latestVersion,
        message: release.body?.slice(0, 200) || `Version ${latestVersion} is available.`,
        downloadUrl: dmgAsset?.browser_download_url || release.html_url,
      });
    }
  } catch {}
}

function showUpdatePopup(update) {
  if (document.getElementById('updatePopup')) return;
  const overlay = document.createElement('div');
  overlay.id = 'updatePopup';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;z-index:9999;';
  const popup = document.createElement('div');
  popup.style.cssText = 'background:white;border-radius:16px;padding:32px;max-width:420px;width:90%;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,0.15);';
  popup.innerHTML = `
    <div style="width:48px;height:48px;margin:0 auto 16px;background:#22c55e;border-radius:50%;display:flex;align-items:center;justify-content:center;">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5"><path d="M12 2v14m0 0l-5-5m5 5l5-5M4 20h16"/></svg>
    </div>
    <h2 style="font-size:20px;font-weight:700;margin:0 0 8px;">AURA ${update.version}</h2>
    <p style="font-size:14px;color:#6b6b7b;margin:0 0 24px;line-height:1.5;">${update.message}</p>
    <div style="display:flex;gap:12px;justify-content:center;">
      <button id="updateLaterBtn" style="padding:10px 20px;border-radius:10px;border:1px solid #d4d4d8;background:white;cursor:pointer;font-size:14px;font-weight:600;">Later</button>
      <button id="updateNowBtn" style="padding:10px 20px;border-radius:10px;border:none;background:#1a1a1a;color:white;cursor:pointer;font-size:14px;font-weight:600;">Update now</button>
    </div>`;
  overlay.appendChild(popup);
  document.body.appendChild(overlay);

  document.getElementById('updateLaterBtn').addEventListener('click', () => overlay.remove());
  document.getElementById('updateNowBtn').addEventListener('click', async () => {
    const btn = document.getElementById('updateNowBtn');
    btn.textContent = 'Downloading...';
    btn.disabled = true;
    if (window.__TAURI__) {
      try {
        const { check } = window.__TAURI__.updater;
        const updateResult = await check();
        if (updateResult?.available) {
          await updateResult.downloadAndInstall();
          await window.__TAURI__.process.relaunch();
        }
      } catch {
        window.open(update.downloadUrl, '_blank');
      }
    } else {
      window.open(update.downloadUrl, '_blank');
    }
  });
}

// Check for updates every 30 minutes + on startup
setInterval(checkForUpdate, 1800000);
setTimeout(checkForUpdate, 10000);
