#!/usr/bin/env node
// ── AURA Agent Server — executes system commands for AURA tool calling ──
// Runs on port 11436, only accepts localhost connections
const http = require('http');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

const PORT = 11436;
const MAX_OUTPUT = 8000; // Max chars per command output

function jsonResponse(res, data, status = 200) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  });
  res.end(JSON.stringify(data));
}

function readBody(req) {
  return new Promise((resolve) => {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => { try { resolve(JSON.parse(body)); } catch { resolve({}); } });
  });
}

const server = http.createServer(async (req, res) => {
  // CORS preflight
  if (req.method === 'OPTIONS') { jsonResponse(res, {}); return; }
  if (req.method !== 'POST') { jsonResponse(res, { error: 'POST only' }, 405); return; }

  const body = await readBody(req);

  // ── Execute command ──
  if (req.url === '/exec') {
    const cmd = body.command;
    if (!cmd) { jsonResponse(res, { error: 'No command provided' }); return; }
    exec(cmd, { timeout: 25000, maxBuffer: 1024 * 1024, shell: '/bin/zsh' }, (err, stdout, stderr) => {
      const output = (stdout || '').slice(0, MAX_OUTPUT) + (stderr ? '\n[stderr] ' + stderr.slice(0, 2000) : '');
      jsonResponse(res, { output: output || (err ? err.message : 'Done'), exit_code: err ? err.code : 0 });
    });
    return;
  }

  // ── Read file ──
  if (req.url === '/read') {
    try {
      const content = fs.readFileSync(body.path, 'utf-8').slice(0, MAX_OUTPUT * 2);
      jsonResponse(res, { content });
    } catch (e) { jsonResponse(res, { error: e.message }); }
    return;
  }

  // ── Write file ──
  if (req.url === '/write') {
    try {
      const dir = path.dirname(body.path);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(body.path, body.content, 'utf-8');
      jsonResponse(res, { success: true });
    } catch (e) { jsonResponse(res, { error: e.message }); }
    return;
  }

  // ── List directory ──
  if (req.url === '/ls') {
    try {
      const entries = fs.readdirSync(body.path).map(f => {
        const full = path.join(body.path, f);
        const stat = fs.statSync(full);
        return (stat.isDirectory() ? '[DIR] ' : '[FILE] ') + f;
      });
      jsonResponse(res, { entries });
    } catch (e) { jsonResponse(res, { error: e.message }); }
    return;
  }

  // ── Extract PDF text ──
  if (req.url === '/extract-pdf') {
    const b64 = body.base64;
    if (!b64) { jsonResponse(res, { error: 'No base64 data' }); return; }
    const tmpFile = path.join(require('os').tmpdir(), `aura_pdf_${Date.now()}.pdf`);
    try {
      fs.writeFileSync(tmpFile, Buffer.from(b64, 'base64'));
      // Try pypdf first, fallback to pdftotext
      exec(`python3 -c "
from pypdf import PdfReader
reader = PdfReader('${tmpFile}')
text = ''
for page in reader.pages:
    text += page.extract_text() + '\\n---\\n'
print(text[:8000])
" 2>/dev/null || pdftotext '${tmpFile}' - 2>/dev/null | head -500 || echo "[Could not extract PDF text]"`,
        { timeout: 15000, maxBuffer: 1024 * 1024 }, (err, stdout) => {
          try { fs.unlinkSync(tmpFile); } catch {}
          jsonResponse(res, { text: (stdout || '').slice(0, 8000) || '[PDF text extraction failed]' });
        });
    } catch (e) {
      try { fs.unlinkSync(tmpFile); } catch {}
      jsonResponse(res, { error: e.message });
    }
    return;
  }

  jsonResponse(res, { error: 'Unknown endpoint' }, 404);
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`AURA Agent Server running on http://127.0.0.1:${PORT}`);
});
