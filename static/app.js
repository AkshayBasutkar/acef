/**
 * ACEF Web Application Controller
 * Handles interactive chat, ingestion studio, health telemetry, and live architecture visualization.
 */

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  sessionId: generateUUID(),
  isQuerying: false,
};

function generateUUID() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// ── DOM Elements ─────────────────────────────────────────────────────────────
const elements = {
  displaySessionId: document.getElementById('display-session-id'),
  btnNewSession: document.getElementById('btn-new-session'),
  pillNeo4j: document.getElementById('pill-neo4j'),
  pillChroma: document.getElementById('pill-chroma'),
  pillLlm: document.getElementById('pill-llm'),
  btnRefreshHealth: document.getElementById('btn-refresh-health'),

  // Tabs
  tabBtns: document.querySelectorAll('.tab-btn'),
  tabContents: document.querySelectorAll('.tab-content'),

  // Ingestion
  inputCodePath: document.getElementById('input-code-path'),
  btnIngestCode: document.getElementById('btn-ingest-code'),
  codeIngestStatus: document.getElementById('code-ingest-status'),
  dropZone: document.getElementById('drop-zone'),
  fileInput: document.getElementById('file-input'),
  btnIngestDemoDoc: document.getElementById('btn-ingest-demo-doc'),
  docIngestStatus: document.getElementById('doc-ingest-status'),
  btnClearCache: document.getElementById('btn-clear-cache'),

  // Visualizer steps & branches
  stepCache: document.getElementById('step-cache'),
  badgeCacheResult: document.getElementById('badge-cache-result'),
  stepClassify: document.getElementById('step-classify'),
  badgeClassifyResult: document.getElementById('badge-classify-result'),
  branchSimple: document.getElementById('branch-simple'),
  branchCode: document.getElementById('branch-code'),
  branchMemory: document.getElementById('branch-memory'),
  stepLlm: document.getElementById('step-llm'),
  badgeLlmResult: document.getElementById('badge-llm-result'),
  stepWriteback: document.getElementById('step-writeback'),
  badgeWritebackResult: document.getElementById('badge-writeback-result'),

  // Chat
  messagesContainer: document.getElementById('messages-container'),
  welcomeBanner: document.getElementById('welcome-banner'),
  queryForm: document.getElementById('query-form'),
  queryInput: document.getElementById('query-input'),
  btnSend: document.getElementById('btn-send'),
  dockStatus: document.getElementById('dock-status'),

  // View Switcher & Panels
  btnModeChat: document.getElementById('btn-mode-chat'),
  btnModeGraph: document.getElementById('btn-mode-graph'),
  btnQuickViewGraph: document.getElementById('btn-quick-view-graph'),
  panelChat: document.getElementById('panel-chat'),
  panelGraph: document.getElementById('panel-graph'),
};

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  updateSessionDisplay();
  checkHealth();
  setupEventListeners();
});

function updateSessionDisplay() {
  elements.displaySessionId.textContent = state.sessionId.substring(0, 13) + '...';
  elements.displaySessionId.title = `Full Session ID: ${state.sessionId}`;
}

// ── Health Check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Neo4j
    updatePill(elements.pillNeo4j, data.neo4j_connected, 'Connected', 'Offline');
    // Chroma
    updatePill(elements.pillChroma, data.chroma_ready, 'Ready', 'Unavailable');
    // LLM
    updatePill(elements.pillLlm, data.llm_configured, 'Active', 'Unconfigured');
  } catch (err) {
    console.error('Health check failed:', err);
    updatePill(elements.pillNeo4j, false, 'Connected', 'Offline');
    updatePill(elements.pillChroma, false, 'Ready', 'Offline');
    updatePill(elements.pillLlm, false, 'Active', 'Offline');
  }
}

function updatePill(pill, isOk, okText, errText) {
  pill.classList.remove('status-ok', 'status-err');
  pill.classList.add(isOk ? 'status-ok' : 'status-err');
  pill.querySelector('.val').textContent = isOk ? okText : errText;
}

// ── Event Listeners ──────────────────────────────────────────────────────────
function setupEventListeners() {
  // Refresh Health
  elements.btnRefreshHealth.addEventListener('click', () => {
    checkHealth();
  });

  // New Session
  elements.btnNewSession.addEventListener('click', () => {
    state.sessionId = generateUUID();
    updateSessionDisplay();
    // Clear chat messages except welcome card
    elements.messagesContainer.innerHTML = '';
    elements.messagesContainer.appendChild(elements.welcomeBanner);
    elements.welcomeBanner.style.display = 'block';
    resetVisualizer();
    setDockStatus('New session started');
  });

  // View Mode Switching (Chat Studio vs Knowledge Graph)
  function switchView(mode) {
    if (mode === 'chat') {
      if (elements.btnModeChat) elements.btnModeChat.classList.add('active');
      if (elements.btnModeGraph) elements.btnModeGraph.classList.remove('active');
      if (elements.panelChat) elements.panelChat.style.display = 'flex';
      if (elements.panelGraph) elements.panelGraph.style.display = 'none';
    } else if (mode === 'graph') {
      if (elements.btnModeGraph) elements.btnModeGraph.classList.add('active');
      if (elements.btnModeChat) elements.btnModeChat.classList.remove('active');
      if (elements.panelChat) elements.panelChat.style.display = 'none';
      if (elements.panelGraph) elements.panelGraph.style.display = 'flex';
      graphVisualizer.init();
    }
  }

  if (elements.btnModeChat) elements.btnModeChat.addEventListener('click', () => switchView('chat'));
  if (elements.btnModeGraph) elements.btnModeGraph.addEventListener('click', () => switchView('graph'));
  if (elements.btnQuickViewGraph) {
    elements.btnQuickViewGraph.addEventListener('click', () => switchView('graph'));
  }

  // Tab Switching
  elements.tabBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      elements.tabBtns.forEach((b) => b.classList.remove('active'));
      elements.tabContents.forEach((c) => c.classList.remove('active'));
      btn.classList.add('active');
      const tabId = `tab-${btn.dataset.tab}`;
      const content = document.getElementById(tabId);
      if (content) content.classList.add('active');
    });
  });

  // Auto-resize Query Input
  elements.queryInput.addEventListener('input', () => {
    elements.queryInput.style.height = 'auto';
    elements.queryInput.style.height = Math.min(elements.queryInput.scrollHeight, 140) + 'px';
  });

  // Enter to send (Shift+Enter for newline)
  elements.queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      elements.queryForm.dispatchEvent(new Event('submit'));
    }
  });

  // Form Submit
  elements.queryForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = elements.queryInput.value.trim();
    if (!query || state.isQuerying) return;
    submitQuery(query);
  });

  // Quick Prompts
  document.querySelectorAll('.prompt-card').forEach((card) => {
    card.addEventListener('click', () => {
      const q = card.dataset.query;
      elements.queryInput.value = q;
      elements.queryForm.dispatchEvent(new Event('submit'));
    });
  });

  // Ingest Code
  elements.btnIngestCode.addEventListener('click', handleCodeIngest);

  // Ingest Document Drop Zone
  elements.dropZone.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleDocUpload(e.target.files[0]);
    }
  });
  elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('dragover');
  });
  elements.dropZone.addEventListener('dragleave', () => {
    elements.dropZone.classList.remove('dragover');
  });
  elements.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleDocUpload(e.dataTransfer.files[0]);
    }
  });

  // Ingest Demo Doc preset
  elements.btnIngestDemoDoc.addEventListener('click', handleDemoDocPreset);

  // Clear Cache
  elements.btnClearCache.addEventListener('click', async () => {
    elements.btnClearCache.disabled = true;
    try {
      // Re-triggering ingest of 0 files or clear
      setDockStatus('Clearing semantic cache...');
      // Re-ingest code clears cache, or we can notify user
      const res = await fetch('/ingest/code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ root_path: './data/demo_codebase' }),
      });
      if (res.ok) {
        setDockStatus('Semantic Cache Cleared & Demo Code Synchronized');
      }
    } catch (err) {
      setDockStatus('Cache clear error: ' + err.message);
    } finally {
      elements.btnClearCache.disabled = false;
    }
  });
}

// ── Ingestion Handlers ────────────────────────────────────────────────────────
async function handleCodeIngest() {
  const rootPath = elements.inputCodePath.value.trim();
  if (!rootPath) return;

  elements.btnIngestCode.disabled = true;
  elements.btnIngestCode.textContent = 'Ingesting...';
  showStatus(elements.codeIngestStatus, 'Parsing AST and building Neo4j graph...', 'info');

  try {
    const res = await fetch('/ingest/code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ root_path: rootPath }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Code ingestion failed');

    const s = data.summary || {};
    const fileCount = s.files ?? s.files_scanned ?? 0;
    const symCount = s.symbols ?? s.symbols_created ?? 0;
    const callCount = s.calls ?? s.call_relationships ?? 0;
    let msg = `Indexed ${fileCount} Python file${fileCount === 1 ? '' : 's'}, ${symCount} symbols, ${callCount} calls.`;
    if (s.skipped_non_python > 0) {
      const extList = s.skipped_extensions && s.skipped_extensions.length > 0
        ? ` (${s.skipped_extensions.join(', ')})`
        : '';
      msg += `\n⚠️ Skipped ${s.skipped_non_python} non-Python file${s.skipped_non_python === 1 ? '' : 's'}${extList}.`;
    }
    showStatus(elements.codeIngestStatus, msg, 'success');
    checkHealth();
    if (graphVisualizer.initialized) {
      graphVisualizer.loadData();
    }
  } catch (err) {
    showStatus(elements.codeIngestStatus, err.message, 'error');
  } finally {
    elements.btnIngestCode.disabled = false;
    elements.btnIngestCode.textContent = 'Ingest';
  }
}

async function handleDocUpload(file) {
  showStatus(elements.docIngestStatus, `Uploading & chunking ${file.name}...`, 'info');
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/ingest/document', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Document ingestion failed');

    showStatus(
      elements.docIngestStatus,
      `Success: ${file.name} chunked into ${data.chunks_created ?? 0} embeddings.`,
      'success'
    );
  } catch (err) {
    showStatus(elements.docIngestStatus, err.message, 'error');
  }
}

async function handleDemoDocPreset() {
  // Fetch local demo document from server and send as blob
  showStatus(elements.docIngestStatus, 'Ingesting demo pricing.md...', 'info');
  try {
    const demoContent = `# ACEF Cloud Platform Pricing

## Plans
- Starter Tier: Free tier, 10,000 monthly queries, community support.
- Pro Tier: $49/mo, 250,000 queries, priority email support.
- Enterprise Tier: Custom pricing, unlimited queries, 99.9% uptime SLA, dedicated technical account manager.

## Refund Policy
Subscriptions may be cancelled within 14 days for a full refund.`;

    const blob = new Blob([demoContent], { type: 'text/markdown' });
    const file = new File([blob], 'pricing.md', { type: 'text/markdown' });
    await handleDocUpload(file);
  } catch (err) {
    showStatus(elements.docIngestStatus, err.message, 'error');
  }
}

function showStatus(elem, msg, type) {
  elem.className = `ingest-status show ${type}`;
  elem.textContent = msg;
}

// ── Query Submission ──────────────────────────────────────────────────────────
async function submitQuery(query) {
  state.isQuerying = true;
  elements.btnSend.disabled = true;
  elements.welcomeBanner.style.display = 'none';

  // Render User Message
  renderUserMessage(query);
  elements.queryInput.value = '';
  elements.queryInput.style.height = 'auto';

  // Render Skeleton Loader
  const skeletonRow = renderSkeletonMessage();
  scrollToBottom();

  setDockStatus('Routing query through ACEF pipeline...');
  resetVisualizer();

  const startTime = performance.now();

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        session_id: state.sessionId,
      }),
    });

    const data = await res.json();
    const clientLatency = Math.round(performance.now() - startTime);

    if (!res.ok) {
      throw new Error(data.detail || `Server error (${res.status})`);
    }

    // Remove skeleton
    skeletonRow.remove();

    // Render Assistant Response
    renderAssistantMessage(data, clientLatency);

    // Update Architecture Visualizer
    updateVisualizer(data);

    setDockStatus(`Response completed in ${data.latency_ms ?? clientLatency} ms`);
  } catch (err) {
    skeletonRow.remove();
    renderErrorMessage(err.message);
    setDockStatus(`Error: ${err.message}`);
  } finally {
    state.isQuerying = false;
    elements.btnSend.disabled = false;
    scrollToBottom();
  }
}

// ── Visualizer Updates ────────────────────────────────────────────────────────
function resetVisualizer() {
  document.querySelectorAll('.flow-step, .branch-card').forEach((el) => {
    el.classList.remove('active-step', 'active-branch');
  });
  elements.badgeCacheResult.textContent = '—';
  elements.badgeClassifyResult.textContent = '—';
  elements.badgeLlmResult.textContent = '—';
  elements.badgeWritebackResult.textContent = '—';
}

function updateVisualizer(resp) {
  resetVisualizer();

  // Step 1: Cache
  elements.stepCache.classList.add('active-step');
  elements.badgeCacheResult.textContent = resp.from_cache ? 'HIT' : 'MISS';

  if (resp.from_cache) {
    elements.badgeCacheResult.style.color = '#f59e0b';
    return; // Bypass subsequent steps on cache hit
  }

  // Step 2: Classifier
  elements.stepClassify.classList.add('active-step');
  const qType = (resp.query_type || 'simple').toUpperCase();
  elements.badgeClassifyResult.textContent = qType;

  // Step 3: Branch
  if (qType === 'SIMPLE') {
    elements.branchSimple.classList.add('active-branch');
  } else if (qType === 'CODE') {
    elements.branchCode.classList.add('active-branch');
  } else if (qType === 'MEMORY') {
    elements.branchMemory.classList.add('active-branch');
  }

  // Step 4: LLM
  elements.stepLlm.classList.add('active-step');
  elements.badgeLlmResult.textContent = `${Math.round(resp.latency_ms || 0)}ms`;

  // Step 5: Write-back
  elements.stepWriteback.classList.add('active-step');
  elements.badgeWritebackResult.textContent = qType === 'SIMPLE' ? 'Cached' : 'Session Memory';
}

// ── Message Renderers ─────────────────────────────────────────────────────────
function renderUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user-row';
  row.innerHTML = `
    <div class="message-avatar">U</div>
    <div class="message-bubble">
      <div class="message-content">${escapeHtml(text)}</div>
    </div>
  `;
  elements.messagesContainer.appendChild(row);
}

function renderSkeletonMessage() {
  const row = document.createElement('div');
  row.className = 'message-row assistant-row skeleton-row';
  row.innerHTML = `
    <div class="message-avatar">⚡</div>
    <div class="message-bubble">
      <div class="skeleton-line" style="width: 40%;"></div>
      <div class="skeleton-line" style="width: 85%;"></div>
      <div class="skeleton-line" style="width: 65%;"></div>
    </div>
  `;
  elements.messagesContainer.appendChild(row);
  return row;
}

function renderAssistantMessage(data, fallbackLatency) {
  const row = document.createElement('div');
  row.className = 'message-row assistant-row';

  const qType = (data.query_type || 'simple').toLowerCase();
  const fromCache = data.from_cache === true;
  const latency = Math.round(data.latency_ms || fallbackLatency);
  const sources = data.sources || [];

  let typeBadgeClass = 'telemetry-type-simple';
  if (qType === 'code') typeBadgeClass = 'telemetry-type-code';
  if (qType === 'memory') typeBadgeClass = 'telemetry-type-memory';

  const cacheBadgeClass = fromCache ? 'telemetry-cache-hit' : 'telemetry-cache-miss';
  const cacheText = fromCache ? '⚡ CACHE HIT' : 'CACHE MISS';

  // Build retrieved sources accordion if any
  let evidenceHtml = '';
  if (sources.length > 0) {
    const itemsHtml = sources
      .map((s, idx) => {
        return `<div class="evidence-item"><strong>[Chunk ${idx + 1}]</strong>\n${escapeHtml(s)}</div>`;
      })
      .join('');

    evidenceHtml = `
      <div class="evidence-drawer">
        <div class="evidence-header" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'block' ? 'none' : 'block'">
          <span>🔍 Ground-Truth Retrieval Evidence (${sources.length} sources)</span>
          <span>▼ Toggle Context</span>
        </div>
        <div class="evidence-content" style="display: none;">
          ${itemsHtml}
        </div>
      </div>
    `;
  }

  row.innerHTML = `
    <div class="message-avatar">⚡</div>
    <div class="message-bubble">
      <div class="telemetry-bar">
        <span class="telemetry-tag ${typeBadgeClass}">${qType.toUpperCase()}</span>
        <span class="telemetry-tag ${cacheBadgeClass}">${cacheText}</span>
        <span class="telemetry-tag telemetry-latency">⏱ ${latency} ms</span>
      </div>
      <div class="message-content">${formatMarkdown(data.response || '')}</div>
      ${evidenceHtml}
    </div>
  `;

  elements.messagesContainer.appendChild(row);
}

function renderErrorMessage(errText) {
  const row = document.createElement('div');
  row.className = 'message-row assistant-row';
  row.innerHTML = `
    <div class="message-avatar" style="background: var(--accent-rose);">!</div>
    <div class="message-bubble" style="border-color: rgba(244, 63, 94, 0.3); background: rgba(244, 63, 94, 0.08);">
      <div class="message-content" style="color: #fda4af;">
        <strong>System Notice:</strong> ${escapeHtml(errText)}
      </div>
    </div>
  `;
  elements.messagesContainer.appendChild(row);
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function scrollToBottom() {
  elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

function setDockStatus(msg) {
  elements.dockStatus.textContent = msg;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatMarkdown(text) {
  if (!text) return '';

  // Escape HTML first
  let out = escapeHtml(text);

  // Fenced Code blocks ```lang\ncode\n```
  out = out.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
  });

  // Inline code `code`
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bold **text**
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // Headings
  out = out.replace(/^### (.*$)/gim, '<h4 style="margin: 8px 0 4px; color: #fff;">$1</h4>');
  out = out.replace(/^## (.*$)/gim, '<h3 style="margin: 10px 0 4px; color: #fff;">$1</h3>');
  out = out.replace(/^# (.*$)/gim, '<h2 style="margin: 12px 0 6px; color: #fff;">$1</h2>');

  // Unordered list items
  out = out.replace(/^\- (.*$)/gim, '<li style="margin-left: 18px;">$1</li>');

  // Convert newlines to breaks where appropriate
  out = out.replace(/\n\n/g, '<p></p>');
  out = out.replace(/\n/g, '<br>');

  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// In-App Knowledge Graph Visualizer Engine (Force-Directed Canvas)
// ─────────────────────────────────────────────────────────────────────────────
const graphVisualizer = {
  initialized: false,
  canvas: null,
  ctx: null,
  dpr: 1,
  width: 800,
  height: 600,

  // Graph Data
  nodes: [],
  links: [],
  nodeMap: new Map(),

  // Simulation & Viewport
  physicsRunning: true,
  animId: null,
  zoom: 1.0,
  panX: 0,
  panY: 0,

  // Interactions
  hoveredNode: null,
  selectedNode: null,
  draggedNode: null,
  isPanning: false,
  startMouseX: 0,
  startMouseY: 0,
  mouseScreenX: 0,
  mouseScreenY: 0,

  // Filters
  filters: {
    file: true,
    function: true,
    class: true,
    CALLS: true,
    DEFINES: true,
    IMPORTS: true,
  },
  searchQuery: '',

  init() {
    this.canvas = document.getElementById('graph-canvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;

    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());

    if (!this.initialized) {
      this.setupControls();
      this.setupCanvasEvents();
      this.loadData();
      this.startLoop();
      this.initialized = true;
    } else {
      this.resizeCanvas();
      if (!this.physicsRunning) {
        this.render();
      }
    }
  },

  resizeCanvas() {
    if (!this.canvas) return;
    const parent = this.canvas.parentElement;
    const rect = parent.getBoundingClientRect();
    this.width = Math.max(rect.width, 300);
    this.height = Math.max(rect.height, 300);

    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.canvas.style.width = this.width + 'px';
    this.canvas.style.height = this.height + 'px';
  },

  async loadData() {
    const loading = document.getElementById('graph-loading');
    if (loading) loading.style.display = 'flex';

    try {
      const res = await fetch('/graph/data');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      this.processData(data);
    } catch (err) {
      console.error('Failed to load graph data:', err);
      if (loading) {
        loading.innerHTML = `<span style="color: #fda4af;">Failed to load graph from Neo4j (${escapeHtml(err.message)})</span>`;
      }
    } finally {
      if (loading && this.nodes.length > 0) {
        loading.style.display = 'none';
      }
    }
  },

  processData(data) {
    const rawNodes = data.nodes || [];
    const rawLinks = data.links || [];

    this.nodeMap.clear();
    this.nodes = [];
    this.links = [];

    let countFiles = 0;
    let countFuncs = 0;
    let countClasses = 0;
    let countCalls = 0;
    let countDefines = 0;
    let countImports = 0;

    // Radius, colors, and layout setup
    const total = rawNodes.length;
    const spreadRadius = Math.max(300, Math.sqrt(total) * 45);

    rawNodes.forEach((rn, i) => {
      const angle = (i / total) * Math.PI * 2;
      const dist = 50 + (i % 7) * (spreadRadius / 7);

      const isFile = rn.type === 'File';
      const isClass = rn.kind === 'class';
      const isFunc = !isFile && !isClass;

      if (isFile) countFiles++;
      else if (isClass) countClasses++;
      else countFuncs++;

      const nodeObj = {
        id: rn.id,
        name: rn.name || 'Unnamed',
        type: rn.type,
        kind: rn.kind || rn.type.toLowerCase(),
        docstring: rn.docstring || '',
        file_path: rn.file_path || '',
        lineno: rn.lineno || 0,
        end_lineno: rn.end_lineno || rn.lineno || 0,
        source: rn.source || '',
        x: Math.cos(angle) * dist + (Math.random() - 0.5) * 40,
        y: Math.sin(angle) * dist + (Math.random() - 0.5) * 40,
        vx: 0,
        vy: 0,
        radius: isFile ? 16 : (isClass ? 13 : 10),
        color: isFile ? '#06b6d4' : (isClass ? '#a855f7' : '#10b981'),
        strokeColor: isFile ? '#67e8f9' : (isClass ? '#d8b4fe' : '#6ee7b7'),
        outgoingCalls: [],
        incomingCalls: [],
        defines: [],
        imports: [],
      };

      this.nodeMap.set(rn.id, nodeObj);
      this.nodes.push(nodeObj);
    });

    rawLinks.forEach((rl) => {
      const srcNode = this.nodeMap.get(rl.source);
      const tgtNode = this.nodeMap.get(rl.target);
      if (srcNode && tgtNode) {
        const linkObj = {
          source: srcNode,
          target: tgtNode,
          type: rl.type,
        };
        this.links.push(linkObj);

        if (rl.type === 'CALLS') {
          countCalls++;
          srcNode.outgoingCalls.push(tgtNode);
          tgtNode.incomingCalls.push(srcNode);
        } else if (rl.type === 'DEFINES') {
          countDefines++;
          srcNode.defines.push(tgtNode);
        } else if (rl.type === 'IMPORTS') {
          countImports++;
          srcNode.imports.push(tgtNode);
        }
      }
    });

    // Update Counter Badges
    const elFiles = document.getElementById('count-files');
    const elFuncs = document.getElementById('count-functions');
    const elClasses = document.getElementById('count-classes');
    const elCalls = document.getElementById('count-calls');
    const elDefines = document.getElementById('count-defines');
    const elImports = document.getElementById('count-imports');

    if (elFiles) elFiles.textContent = countFiles;
    if (elFuncs) elFuncs.textContent = countFuncs;
    if (elClasses) elClasses.textContent = countClasses;
    if (elCalls) elCalls.textContent = countCalls;
    if (elDefines) elDefines.textContent = countDefines;
    if (elImports) elImports.textContent = countImports;

    // Reset View to Center
    this.resetView();
  },

  resetView() {
    this.zoom = 0.85;
    this.panX = 0;
    this.panY = 0;
  },

  startLoop() {
    const loop = () => {
      if (this.physicsRunning) {
        this.updatePhysics();
      }
      this.render();
      this.animId = requestAnimationFrame(loop);
    };
    this.animId = requestAnimationFrame(loop);
  },

  updatePhysics() {
    const visibleNodes = this.nodes.filter((n) => this.isNodeVisible(n));
    const count = visibleNodes.length;
    if (count === 0) return;

    // 1. Repulsion between visible node pairs
    for (let i = 0; i < count; i++) {
      const n1 = visibleNodes[i];
      for (let j = i + 1; j < count; j++) {
        const n2 = visibleNodes[j];
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const distSq = dx * dx + dy * dy || 1;
        if (distSq < 250000) { // Limit repulsion radius to ~500px for speed
          const dist = Math.sqrt(distSq);
          const force = 1400 / (distSq + 50);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          n1.vx -= fx;
          n1.vy -= fy;
          n2.vx += fx;
          n2.vy += fy;
        }
      }
    }

    // 2. Spring attraction along links
    this.links.forEach((l) => {
      if (!this.isLinkVisible(l)) return;
      const s = l.source;
      const t = l.target;
      const dx = t.x - s.x;
      const dy = t.y - s.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const targetDist = l.type === 'DEFINES' ? 65 : (l.type === 'CALLS' ? 100 : 130);
      const force = (dist - targetDist) * 0.035;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;

      s.vx += fx;
      s.vy += fy;
      t.vx -= fx;
      t.vy -= fy;
    });

    // 3. Center gravity pull & velocity damping
    visibleNodes.forEach((n) => {
      if (n === this.draggedNode) return; // Don't move if dragging

      n.vx += (0 - n.x) * 0.0012;
      n.vy += (0 - n.y) * 0.0012;

      n.x += n.vx;
      n.y += n.vy;

      n.vx *= 0.82;
      n.vy *= 0.82;
    });
  },

  isNodeVisible(n) {
    if (n.type === 'File' && !this.filters.file) return false;
    if (n.kind === 'function' && !this.filters.function) return false;
    if (n.kind === 'class' && !this.filters.class) return false;
    return true;
  },

  isLinkVisible(l) {
    if (!this.isNodeVisible(l.source) || !this.isNodeVisible(l.target)) return false;
    if (l.type === 'CALLS' && !this.filters.CALLS) return false;
    if (l.type === 'DEFINES' && !this.filters.DEFINES) return false;
    if (l.type === 'IMPORTS' && !this.filters.IMPORTS) return false;
    return true;
  },

  render() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const width = this.width;
    const height = this.height;

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Apply High-DPI, Center offset, Pan, and Zoom
    ctx.scale(this.dpr, this.dpr);
    ctx.translate(width / 2 + this.panX, height / 2 + this.panY);
    ctx.scale(this.zoom, this.zoom);

    // Connected neighbors set for focus highlighting
    const highlightNode = this.hoveredNode || this.selectedNode;
    const connectedSet = new Set();
    if (highlightNode) {
      connectedSet.add(highlightNode.id);
      this.links.forEach((l) => {
        if (l.source.id === highlightNode.id) connectedSet.add(l.target.id);
        if (l.target.id === highlightNode.id) connectedSet.add(l.source.id);
      });
    }

    // ── 1. Draw Links ─────────────────────────────────────────────────────────
    this.links.forEach((l) => {
      if (!this.isLinkVisible(l)) return;

      const isAttached = highlightNode && (l.source.id === highlightNode.id || l.target.id === highlightNode.id);
      const isDimmed = highlightNode && !isAttached;

      ctx.beginPath();
      ctx.moveTo(l.source.x, l.source.y);
      ctx.lineTo(l.target.x, l.target.y);

      if (l.type === 'CALLS') {
        ctx.strokeStyle = isAttached ? '#34d399' : (isDimmed ? 'rgba(16, 185, 129, 0.08)' : 'rgba(16, 185, 129, 0.35)');
        ctx.lineWidth = isAttached ? 2.5 : 1.2;
        ctx.setLineDash([]);
      } else if (l.type === 'DEFINES') {
        ctx.strokeStyle = isAttached ? '#67e8f9' : (isDimmed ? 'rgba(6, 182, 212, 0.06)' : 'rgba(6, 182, 212, 0.25)');
        ctx.lineWidth = isAttached ? 2.0 : 1.0;
        ctx.setLineDash([]);
      } else if (l.type === 'IMPORTS') {
        ctx.strokeStyle = isAttached ? '#c4b5fd' : (isDimmed ? 'rgba(139, 92, 246, 0.06)' : 'rgba(139, 92, 246, 0.28)');
        ctx.lineWidth = isAttached ? 2.0 : 1.0;
        ctx.setLineDash([4, 4]);
      }
      ctx.stroke();

      // Draw directional arrow on CALLS edges
      if (l.type === 'CALLS' && (!isDimmed || isAttached)) {
        this.drawArrow(ctx, l.source.x, l.source.y, l.target.x, l.target.y, l.target.radius, ctx.strokeStyle);
      }
    });

    ctx.setLineDash([]); // Reset line dash

    // ── 2. Draw Nodes ─────────────────────────────────────────────────────────
    const query = this.searchQuery.toLowerCase().trim();

    this.nodes.forEach((n) => {
      if (!this.isNodeVisible(n)) return;

      const isHovered = n === this.hoveredNode;
      const isSelected = n === this.selectedNode;
      const isConnected = highlightNode ? connectedSet.has(n.id) : true;
      const isMatch = query && n.name.toLowerCase().includes(query);

      ctx.save();
      ctx.globalAlpha = (!highlightNode || isConnected || isMatch) ? 1.0 : 0.18;

      // Glowing Halo for hover / selection / search match
      if (isHovered || isSelected || isMatch) {
        ctx.shadowColor = isMatch ? '#f59e0b' : n.strokeColor;
        ctx.shadowBlur = isMatch ? 24 : 18;
      }

      // Outer border circle
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius + (isHovered ? 3 : (isSelected ? 2 : 0)), 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.fill();

      ctx.lineWidth = isHovered || isSelected ? 2.5 : 1.5;
      ctx.strokeStyle = isMatch ? '#f59e0b' : n.strokeColor;
      ctx.stroke();

      // Node Label (rendered crisp with pill background)
      if (this.zoom >= 0.65 || isHovered || isSelected || isMatch || n.type === 'File') {
        this.drawNodeLabel(ctx, n, isHovered || isSelected || isMatch);
      }

      ctx.restore();
    });

    ctx.restore();
  },

  drawArrow(ctx, fromX, fromY, toX, toY, targetRadius, color) {
    const dx = toX - fromX;
    const dy = toY - fromY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < targetRadius + 15) return;

    // Arrow position slightly outside node circle
    const arrowDist = dist - (targetRadius + 4);
    const arrowX = fromX + (dx / dist) * arrowDist;
    const arrowY = fromY + (dy / dist) * arrowDist;
    const angle = Math.atan2(dy, dx);
    const headLen = 7;

    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(
      arrowX - headLen * Math.cos(angle - Math.PI / 6),
      arrowY - headLen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      arrowX - headLen * Math.cos(angle + Math.PI / 6),
      arrowY - headLen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  },

  drawNodeLabel(ctx, n, isEmphasized) {
    const fontSize = isEmphasized ? 11 : 9.5;
    ctx.font = `${isEmphasized ? '600' : '500'} ${fontSize}px "JetBrains Mono", monospace`;
    const label = n.name.length > 24 ? n.name.substring(0, 22) + '…' : n.name;
    const textWidth = ctx.measureText(label).width;

    const labelX = n.x;
    const labelY = n.y + n.radius + 12;

    // Background badge pill for readability
    ctx.fillStyle = isEmphasized ? 'rgba(15, 23, 42, 0.95)' : 'rgba(10, 15, 26, 0.8)';
    ctx.beginPath();
    ctx.roundRect(labelX - textWidth / 2 - 4, labelY - fontSize + 1, textWidth + 8, fontSize + 5, 4);
    ctx.fill();
    ctx.strokeStyle = isEmphasized ? n.strokeColor : 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Text label
    ctx.fillStyle = isEmphasized ? '#fff' : '#cbd5e1';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, labelX, labelY - 1);
  },

  // ── Coordinates and Picking ─────────────────────────────────────────────────
  screenToWorld(screenX, screenY) {
    const centerOffsetX = this.width / 2 + this.panX;
    const centerOffsetY = this.height / 2 + this.panY;
    const worldX = (screenX - centerOffsetX) / this.zoom;
    const worldY = (screenY - centerOffsetY) / this.zoom;
    return { x: worldX, y: worldY };
  },

  findNodeAt(screenX, screenY) {
    const { x, y } = this.screenToWorld(screenX, screenY);
    // Search in reverse so top-drawn nodes take precedence
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      if (!this.isNodeVisible(n)) continue;
      const dx = n.x - x;
      const dy = n.y - y;
      const hitRadius = Math.max(n.radius + 4, 16 / this.zoom);
      if (dx * dx + dy * dy <= hitRadius * hitRadius) {
        return n;
      }
    }
    return null;
  },

  // ── Canvas Mouse & Touch Events ─────────────────────────────────────────────
  setupCanvasEvents() {
    const canvas = this.canvas;
    const viewport = document.getElementById('graph-viewport');

    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const hit = this.findNodeAt(mouseX, mouseY);
      if (hit) {
        this.draggedNode = hit;
        this.draggedNode.vx = 0;
        this.draggedNode.vy = 0;
      } else {
        this.isPanning = true;
        this.startMouseX = e.clientX;
        this.startMouseY = e.clientY;
        if (viewport) viewport.classList.add('panning');
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      this.mouseScreenX = mouseX;
      this.mouseScreenY = mouseY;

      if (this.draggedNode) {
        const world = this.screenToWorld(mouseX, mouseY);
        this.draggedNode.x = world.x;
        this.draggedNode.y = world.y;
        this.draggedNode.vx = 0;
        this.draggedNode.vy = 0;
      } else if (this.isPanning) {
        const dx = e.clientX - this.startMouseX;
        const dy = e.clientY - this.startMouseY;
        this.panX += dx;
        this.panY += dy;
        this.startMouseX = e.clientX;
        this.startMouseY = e.clientY;
      } else {
        const hovered = this.findNodeAt(mouseX, mouseY);
        this.hoveredNode = hovered;
        canvas.style.cursor = hovered ? 'pointer' : 'grab';
      }
    });

    window.addEventListener('mouseup', () => {
      if (this.draggedNode) {
        this.draggedNode = null;
      }
      if (this.isPanning) {
        this.isPanning = false;
        if (viewport) viewport.classList.remove('panning');
      }
    });

    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const hit = this.findNodeAt(mouseX, mouseY);
      if (hit) {
        this.selectNode(hit);
      }
    });

    // Zooming with mouse wheel centered at cursor
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      const newZoom = Math.max(0.15, Math.min(4.0, this.zoom * zoomFactor));

      // Zoom towards mouse pointer
      const centerOffsetX = this.width / 2 + this.panX;
      const centerOffsetY = this.height / 2 + this.panY;
      const worldX = (mouseX - centerOffsetX) / this.zoom;
      const worldY = (mouseY - centerOffsetY) / this.zoom;

      this.zoom = newZoom;
      this.panX = mouseX - this.width / 2 - worldX * this.zoom;
      this.panY = mouseY - this.height / 2 - worldY * this.zoom;
    }, { passive: false });
  },

  // ── Node Inspector Drawer ───────────────────────────────────────────────────
  selectNode(node) {
    this.selectedNode = node;
    const inspector = document.getElementById('node-inspector');
    if (!inspector) return;

    // Fill Inspector Elements
    const elType = document.getElementById('inspector-type');
    const elName = document.getElementById('inspector-name');
    const elFile = document.getElementById('inspector-file');
    const elLines = document.getElementById('inspector-lines');
    const rowLines = document.getElementById('row-lineno');

    const secDocstring = document.getElementById('section-docstring');
    const elDocstring = document.getElementById('inspector-docstring');

    const secSource = document.getElementById('section-source');
    const elCode = document.getElementById('inspector-code-content');

    const groupCalls = document.getElementById('group-calls');
    const chipsCalls = document.getElementById('chips-calls');

    const groupCalledBy = document.getElementById('group-called-by');
    const chipsCalledBy = document.getElementById('chips-called-by');

    const groupDefines = document.getElementById('group-defines');
    const chipsDefines = document.getElementById('chips-defines');

    const groupImports = document.getElementById('group-imports');
    const chipsImports = document.getElementById('chips-imports');

    // Type Badge & Name
    elType.className = `inspector-type-badge type-${node.kind}`;
    elType.textContent = node.type === 'File' ? 'File' : (node.kind === 'class' ? 'Class' : 'Function');
    elName.textContent = node.name;

    // File and lines
    elFile.textContent = node.file_path || node.name;
    if (node.lineno > 0) {
      if (rowLines) rowLines.style.display = 'flex';
      elLines.textContent = `L${node.lineno} – L${node.end_lineno}`;
    } else {
      if (rowLines) rowLines.style.display = 'none';
    }

    // Docstring
    if (node.docstring) {
      secDocstring.style.display = 'block';
      elDocstring.textContent = node.docstring;
    } else {
      secDocstring.style.display = 'none';
    }

    // Source Code
    if (node.source) {
      secSource.style.display = 'block';
      elCode.textContent = node.source;
    } else {
      secSource.style.display = 'none';
    }

    // Connected Relationships Chips
    const populateChips = (container, group, items) => {
      container.innerHTML = '';
      if (!items || items.length === 0) {
        group.style.display = 'none';
        return;
      }
      group.style.display = 'block';
      items.forEach((itemNode) => {
        const chip = document.createElement('button');
        chip.className = 'rel-chip';
        chip.textContent = itemNode.name;
        chip.addEventListener('click', () => {
          this.focusOnNode(itemNode);
          this.selectNode(itemNode);
        });
        container.appendChild(chip);
      });
    };

    populateChips(chipsCalls, groupCalls, node.outgoingCalls);
    populateChips(chipsCalledBy, groupCalledBy, node.incomingCalls);
    populateChips(chipsDefines, groupDefines, node.defines);
    populateChips(chipsImports, groupImports, node.imports);

    inspector.style.display = 'flex';
  },

  focusOnNode(node) {
    this.panX = -node.x * this.zoom;
    this.panY = -node.y * this.zoom;
    this.selectedNode = node;
  },

  // ── Toolbar Controls Setup ──────────────────────────────────────────────────
  setupControls() {
    // Close Inspector
    const btnClose = document.getElementById('btn-close-inspector');
    if (btnClose) {
      btnClose.addEventListener('click', () => {
        const inspector = document.getElementById('node-inspector');
        if (inspector) inspector.style.display = 'none';
        this.selectedNode = null;
      });
    }

    // Zoom Buttons
    const btnZoomIn = document.getElementById('btn-graph-zoom-in');
    const btnZoomOut = document.getElementById('btn-graph-zoom-out');
    const btnReset = document.getElementById('btn-graph-reset');

    if (btnZoomIn) {
      btnZoomIn.addEventListener('click', () => {
        this.zoom = Math.min(4.0, this.zoom * 1.25);
      });
    }
    if (btnZoomOut) {
      btnZoomOut.addEventListener('click', () => {
        this.zoom = Math.max(0.15, this.zoom / 1.25);
      });
    }
    if (btnReset) {
      btnReset.addEventListener('click', () => {
        this.resetView();
      });
    }

    // Physics Toggle
    if (this.btnPhysics) {
      this.btnPhysics.addEventListener('click', () => {
        this.physicsRunning = !this.physicsRunning;
        this.btnPhysics.textContent = this.physicsRunning ? 'Pause Physics' : 'Resume Physics';
      });
    }

    // Refresh Graph Data from Neo4j
    const btnRefresh = document.getElementById('btn-graph-refresh');
    if (btnRefresh) {
      btnRefresh.addEventListener('click', () => {
        this.loadData();
      });
    }

    // Filter Chips
    document.querySelectorAll('.filter-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const filterType = chip.dataset.filter;
        const relType = chip.dataset.rel;

        if (filterType) {
          this.filters[filterType] = !this.filters[filterType];
          chip.classList.toggle('active', this.filters[filterType]);
        } else if (relType) {
          this.filters[relType] = !this.filters[relType];
          chip.classList.toggle('active', this.filters[relType]);
        }
      });
    });

    // Search Input
    if (this.searchInput) {
      this.searchInput.addEventListener('input', (e) => {
        this.searchQuery = e.target.value.trim();
        if (this.searchClearBtn) {
          this.searchClearBtn.style.display = this.searchQuery ? 'block' : 'none';
        }
      });

      this.searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          const query = this.searchQuery.toLowerCase();
          const match = this.nodes.find((n) => this.isNodeVisible(n) && n.name.toLowerCase().includes(query));
          if (match) {
            this.focusOnNode(match);
            this.selectNode(match);
          }
        }
      });
    }

    if (this.searchClearBtn) {
      this.searchClearBtn.addEventListener('click', () => {
        this.searchInput.value = '';
        this.searchQuery = '';
        this.searchClearBtn.style.display = 'none';
      });
    }
  },
};
