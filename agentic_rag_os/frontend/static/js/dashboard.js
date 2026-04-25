/* Dashboard JavaScript — Agentic RAG OS */

const API = '/api/v1';
let token = null;
let currentUser = null;
let currentProjectId = null;
let ruleTypes = {};

// ── Token management ──
function getToken() {
  return localStorage.getItem('ragas_token');
}

function authHeaders() {
  return { 'Authorization': `Bearer ${getToken()}`, 'Content-Type': 'application/json' };
}

async function apiGet(path) {
  const res = await fetch(API + path, { headers: authHeaders() });
  if (res.status === 401) { logout(); return null; }
  if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(API + path, { method: 'POST', headers: authHeaders(), body: JSON.stringify(body) });
  if (res.status === 401) { logout(); return null; }
  if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
  return res.json();
}

async function apiDelete(path) {
  const res = await fetch(API + path, { method: 'DELETE', headers: authHeaders() });
  if (res.status === 401) { logout(); return null; }
  if (res.status === 204) return null;
  if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
  return res.json();
}

// ── Toast notifications ──
function toast(msg, type = 'info') {
  const container = document.getElementById('toastContainer');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icons = { success: '✓', error: '✗', info: 'ℹ' };
  el.innerHTML = `<span style="color:var(--${type === 'success' ? 'green' : type === 'error' ? 'pink' : 'indigo'})">${icons[type] || 'ℹ'}</span> ${escHtml(msg)}`;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.4s'; setTimeout(() => el.remove(), 400); }, 3500);
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Auth ──
function logout() {
  localStorage.removeItem('ragas_token');
  window.location.href = '/';
}

async function loadUser() {
  const t = getToken();
  if (!t) { window.location.href = '/'; return; }
  try {
    currentUser = await apiGet('/users/me');
    if (!currentUser) return;
    // Update sidebar
    const av = document.getElementById('sidebarAvatar');
    if (currentUser.avatar_url) {
      av.innerHTML = `<img src="${currentUser.avatar_url}" alt="avatar" />`;
    } else {
      av.textContent = (currentUser.username || '?')[0].toUpperCase();
    }
    document.getElementById('sidebarUsername').textContent = currentUser.display_name || currentUser.username;
    document.getElementById('sidebarTier').textContent = currentUser.tier || 'free';
    document.getElementById('stat-tier').textContent = currentUser.tier || 'free';
    // Storage
    const storage = await apiGet('/users/me/storage');
    if (storage) {
      document.getElementById('stat-storage').textContent = '0 MB';
      document.getElementById('stat-storage-sub').textContent = `of ${storage.limit_mb} MB limit`;
    }
  } catch (e) {
    toast('Failed to load user: ' + e.message, 'error');
  }
}

// ── Page routing ──
function switchPage(page, el) {
  document.querySelectorAll('.page').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
  const pageEl = document.getElementById('page-' + page);
  if (pageEl) pageEl.style.display = '';
  if (el) el.classList.add('active');
  else {
    const btn = document.querySelector(`.sidebar-item[data-page="${page}"]`);
    if (btn) btn.classList.add('active');
  }
  // Load page data
  if (page === 'overview') loadOverview();
  else if (page === 'projects') loadProjects();
  else if (page === 'rewards') loadRewards();
  else if (page === 'compute') loadComputePage();
}

function switchTab(tab, el) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}

// ── Overview ──
async function loadOverview() {
  try {
    const [projects, rewards] = await Promise.all([
      apiGet('/projects'),
      apiGet('/rewards'),
    ]);
    if (projects) {
      document.getElementById('stat-projects').textContent = projects.length;
      const el = document.getElementById('recentProjects');
      if (projects.length === 0) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📂</div><h3>No projects yet</h3><p>Create your first RAG project</p></div>`;
      } else {
        el.innerHTML = projects.slice(0, 4).map(p => projectItem(p)).join('');
      }
    }
    if (rewards) {
      document.getElementById('stat-rewards').textContent = rewards.length;
      const el = document.getElementById('recentRewards');
      if (rewards.length === 0) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">⭐</div><h3>No reward configs yet</h3><p>Define your first reward function</p></div>`;
      } else {
        el.innerHTML = rewards.slice(0, 4).map(r => rewardItem(r)).join('');
      }
    }
  } catch (e) { toast(e.message, 'error'); }
}

// ── Projects ──
async function loadProjects() {
  const el = document.getElementById('projectsList');
  el.innerHTML = `<div style="display:flex;justify-content:center;padding:48px"><div class="spinner"></div></div>`;
  try {
    const projects = await apiGet('/projects');
    if (!projects) return;
    if (projects.length === 0) {
      el.innerHTML = `<div class="empty-state" style="padding:80px;"><div class="empty-state-icon">📂</div><h3>No projects yet</h3><p>Create a RAG project to start ingesting documents</p></div>`;
    } else {
      el.innerHTML = `<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:20px;">${projects.map(p => projectCard(p)).join('')}</div>`;
    }
  } catch (e) { toast(e.message, 'error'); el.innerHTML = `<div class="empty-state"><p style="color:var(--pink);">Error: ${escHtml(e.message)}</p></div>`; }
}

function projectItem(p) {
  return `<div class="project-item">
    <div class="item-icon icon-purple"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375"/></svg></div>
    <div class="item-info"><div class="item-name">${escHtml(p.name)}</div><div class="item-meta">${p.doc_count} docs · ${formatBytes(p.total_bytes)}</div></div>
    <button class="btn btn-ghost" style="padding:6px 12px;font-size:0.78rem;" onclick="openProject(${p.id},'${escHtml(p.name)}','${escHtml(p.description||'')}')">Open</button>
  </div>`;
}

function projectCard(p) {
  return `<div class="panel glass" style="padding:24px;cursor:pointer;" onclick="openProject(${p.id},'${escHtml(p.name)}','${escHtml(p.description||'')}')">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
      <div class="item-icon icon-purple"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375"/></svg></div>
      <div class="item-name" style="font-size:0.95rem;">${escHtml(p.name)}</div>
    </div>
    <p style="color:var(--text3);font-size:0.82rem;margin-bottom:14px;min-height:32px;">${escHtml(p.description||'No description')}</p>
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:var(--text3);">
      <span>${p.doc_count} chunks indexed</span>
      <span>${formatBytes(p.total_bytes)}</span>
    </div>
  </div>`;
}

function openProject(id, name, desc) {
  currentProjectId = id;
  document.getElementById('projectDetailName').textContent = name;
  document.getElementById('projectDetailDesc').textContent = desc || '';
  switchPage('project-detail');
  loadDocuments();
  // Setup drag-drop
  setupUploadDragDrop();
}

function setupUploadDragDrop() {
  const area = document.getElementById('uploadArea');
  if (!area) return;
  area.addEventListener('dragover', e => { e.preventDefault(); area.classList.add('dragover'); });
  area.addEventListener('dragleave', () => area.classList.remove('dragover'));
  area.addEventListener('drop', e => {
    e.preventDefault();
    area.classList.remove('dragover');
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });
}

async function loadDocuments() {
  const el = document.getElementById('documentsList');
  el.innerHTML = `<div style="display:flex;justify-content:center;padding:24px"><div class="spinner"></div></div>`;
  try {
    const docs = await apiGet(`/projects/${currentProjectId}/documents`);
    if (!docs) return;
    if (docs.length === 0) {
      el.innerHTML = `<div class="empty-state" style="padding:40px;"><div class="empty-state-icon">📄</div><h3>No documents yet</h3><p>Upload text files to build your knowledge base</p></div>`;
    } else {
      el.innerHTML = `<div class="panel glass" style="padding:0;">${docs.map(d => `
        <div class="project-item" style="padding:12px 20px;">
          <div class="item-icon icon-blue" style="font-size:1rem;">📄</div>
          <div class="item-info"><div class="item-name">${escHtml(d.filename)}</div><div class="item-meta">${d.doc_count} chunks · ${formatBytes(d.size_bytes)}</div></div>
          <span class="item-badge badge-active">Indexed</span>
        </div>`).join('')}
      </div>`;
    }
  } catch(e) { toast(e.message, 'error'); }
}

async function uploadFiles(files) {
  for (const file of files) {
    const formData = new FormData();
    formData.append('file', file);
    toast(`Uploading ${file.name}…`, 'info');
    try {
      const res = await fetch(`${API}/projects/${currentProjectId}/documents`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${getToken()}` },
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      toast(`${file.name} — ${data.chunks_indexed} chunks indexed`, 'success');
      loadDocuments();
    } catch(e) { toast(`Failed: ${e.message}`, 'error'); }
  }
}

async function runQuery() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;
  const el = document.getElementById('queryResults');
  el.innerHTML = `<div style="display:flex;justify-content:center;padding:24px"><div class="spinner"></div></div>`;
  try {
    const data = await apiPost(`/projects/${currentProjectId}/query`, { query, top_k: 5 });
    if (!data) return;
    if (!data.results || data.results.length === 0) {
      el.innerHTML = `<div class="empty-state" style="margin-top:24px;"><p style="color:var(--text3);">No results found. Try uploading documents first.</p></div>`;
    } else {
      el.innerHTML = data.results.map((r, i) => `
        <div class="panel glass" style="padding:16px 20px;margin-top:12px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:0.75rem;color:var(--text3);">Rank ${r.rank+1} · <span style="color:var(--text2);">${escHtml(r.source||'')}</span></span>
            <span style="font-size:0.75rem;color:var(--green);">Score: ${r.score.toFixed(3)}</span>
          </div>
          <p style="font-size:0.85rem;line-height:1.7;color:var(--text2);">${escHtml(r.content)}</p>
        </div>`).join('');
    }
  } catch(e) { toast(e.message, 'error'); el.innerHTML = ''; }
}

async function createProject() {
  const name = document.getElementById('newProjectName').value.trim();
  const desc = document.getElementById('newProjectDesc').value.trim();
  if (!name) { toast('Project name is required', 'error'); return; }
  try {
    await apiPost('/projects', { name, description: desc || null });
    toast('Project created', 'success');
    closeModal('createProjectModal');
    document.getElementById('newProjectName').value = '';
    document.getElementById('newProjectDesc').value = '';
    loadProjects();
  } catch(e) { toast(e.message, 'error'); }
}

// ── Rewards ──
async function loadRewards() {
  const el = document.getElementById('rewardsList');
  el.innerHTML = `<div style="display:flex;justify-content:center;padding:48px"><div class="spinner"></div></div>`;
  try {
    const configs = await apiGet('/rewards');
    if (!configs) return;
    if (configs.length === 0) {
      el.innerHTML = `<div class="empty-state" style="padding:80px;"><div class="empty-state-icon">⭐</div><h3>No reward configs</h3><p>Create a reward config to start computing RL training signals</p></div>`;
    } else {
      el.innerHTML = `<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:20px;">${configs.map(c => rewardCard(c)).join('')}</div>`;
    }
  } catch(e) { toast(e.message, 'error'); }
}

function rewardItem(c) {
  return `<div class="reward-item">
    <div class="item-icon icon-cyan"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.562.562 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z"/></svg></div>
    <div class="item-info"><div class="item-name">${escHtml(c.name)}</div><div class="item-meta">${c.algorithm.toUpperCase()} · ${(c.rules||[]).length} rules · ${c.compute_count} runs</div></div>
    <span class="item-badge ${c.status==='active'?'badge-active':'badge-draft'}">${c.status}</span>
  </div>`;
}

function rewardCard(c) {
  const rulesCount = (c.rules||[]).length;
  return `<div class="panel glass" style="padding:24px;">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px;">
      <div class="item-name" style="font-size:0.95rem;">${escHtml(c.name)}</div>
      <span class="item-badge ${c.status==='active'?'badge-active':'badge-draft'}">${c.status}</span>
    </div>
    <p style="color:var(--text3);font-size:0.82rem;margin-bottom:14px;min-height:24px;">${escHtml(c.description||'')}</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">
      <span class="tag">${c.algorithm.toUpperCase()}</span>
      <span class="tag">${rulesCount} rule${rulesCount!==1?'s':''}</span>
      <span class="tag">${c.compute_count} run${c.compute_count!==1?'s':''}</span>
    </div>
    <div style="display:flex;gap:8px;">
      <button class="btn btn-primary" style="flex:1;padding:8px;font-size:0.8rem;" onclick="openComputeFor(${c.id})">Compute</button>
      <button class="btn btn-ghost" style="padding:8px 12px;font-size:0.8rem;" onclick="deleteRewardConfig(${c.id})">✕</button>
    </div>
  </div>`;
}

function openComputeFor(configId) {
  switchPage('compute');
  setTimeout(() => {
    const sel = document.getElementById('computeConfigSelect');
    if (sel) sel.value = configId;
  }, 200);
}

async function loadComputePage() {
  try {
    const configs = await apiGet('/rewards');
    if (!configs) return;
    const sel = document.getElementById('computeConfigSelect');
    sel.innerHTML = '<option value="">Select a reward config…</option>' +
      configs.map(c => `<option value="${c.id}">${escHtml(c.name)} (${c.algorithm.toUpperCase()})</option>`).join('');
  } catch(e) { toast(e.message, 'error'); }
}

async function runCompute() {
  const configId = document.getElementById('computeConfigSelect').value;
  const raw = document.getElementById('computeInputs').value.trim();
  if (!configId) { toast('Select a reward config first', 'error'); return; }
  if (!raw) { toast('Add input pairs', 'error'); return; }
  let inputs;
  try { inputs = JSON.parse(raw); }
  catch(e) { toast('Invalid JSON: ' + e.message, 'error'); return; }
  if (!Array.isArray(inputs)) { toast('Input must be a JSON array', 'error'); return; }

  const btn = document.getElementById('computeBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Computing…';
  try {
    const result = await apiPost(`/rewards/${configId}/compute`, { inputs });
    if (!result) return;
    showComputeResults(result);
    toast(`Computed ${result.input_count} rewards — avg: ${result.avg_reward}`, 'success');
  } catch(e) { toast(e.message, 'error'); }
  finally { btn.disabled = false; btn.textContent = 'Compute Rewards'; }
}

function showComputeResults(result) {
  const el = document.getElementById('computeResults');
  const rows = result.results.slice(0, 20);
  el.innerHTML = `
    <div class="panel glass" style="padding:28px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:12px;">
        <div>
          <h3 style="font-size:1rem;font-weight:700;">Computation Results</h3>
          <p style="font-size:0.82rem;color:var(--text3);">${result.input_count} pairs · Avg reward: <strong style="color:var(--green);">${result.avg_reward}</strong></p>
        </div>
        <div style="display:flex;gap:8px;">
          <a href="${API}/rewards/${result.config_id}/computations/${result.id}/export?format=json" class="btn btn-ghost" style="font-size:0.8rem;padding:6px 14px;" download>JSON</a>
          <a href="${API}/rewards/${result.config_id}/computations/${result.id}/export?format=csv" class="btn btn-ghost" style="font-size:0.8rem;padding:6px 14px;" download>CSV</a>
          <a href="${API}/rewards/${result.config_id}/computations/${result.id}/export?format=hf_dataset" class="btn btn-primary" style="font-size:0.8rem;padding:6px 14px;" download>HF Dataset</a>
        </div>
      </div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
          <thead><tr style="border-bottom:1px solid var(--border);">
            <th style="text-align:left;padding:8px 12px;color:var(--text3);font-weight:600;">#</th>
            <th style="text-align:left;padding:8px 12px;color:var(--text3);font-weight:600;">Prompt</th>
            <th style="text-align:left;padding:8px 12px;color:var(--text3);font-weight:600;">Response</th>
            <th style="text-align:right;padding:8px 12px;color:var(--text3);font-weight:600;">Reward</th>
          </tr></thead>
          <tbody>${rows.map((r, i) => `
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
              <td style="padding:8px 12px;color:var(--text3);">${i+1}</td>
              <td style="padding:8px 12px;max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(r.prompt)}</td>
              <td style="padding:8px 12px;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(r.response)}</td>
              <td style="padding:8px 12px;text-align:right;font-weight:700;color:${r.reward > 0.6 ? 'var(--green)' : r.reward > 0.3 ? 'var(--orange)' : 'var(--pink)'};">${r.reward.toFixed(4)}</td>
            </tr>`).join('')}
          </tbody>
        </table>
        ${result.input_count > 20 ? `<p style="text-align:center;color:var(--text3);font-size:0.78rem;padding:12px;">Showing 20 of ${result.input_count}. Export for full results.</p>` : ''}
      </div>
    </div>`;
}

// ── Create Reward Config ──
async function loadRuleTypes() {
  if (Object.keys(ruleTypes).length > 0) return;
  try {
    const data = await apiGet('/rewards/rule-types');
    ruleTypes = data.rule_types || {};
  } catch(e) { /* non-critical */ }
}

function addRule() {
  const container = document.getElementById('rulesContainer');
  const id = Date.now();
  const typeOptions = Object.keys(ruleTypes).length
    ? Object.entries(ruleTypes).map(([k, v]) => `<option value="${k}">${k}</option>`).join('')
    : ['keyword_match','length_range','format_check','no_hallucination','citation_present','reasoning_steps','custom_regex'].map(t => `<option value="${t}">${t}</option>`).join('');

  const div = document.createElement('div');
  div.id = `rule-${id}`;
  div.style.cssText = 'background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:10px;';
  div.innerHTML = `
    <div style="display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap;">
      <div style="flex:1;min-width:140px;">
        <label class="form-label" style="font-size:0.75rem;">Rule name</label>
        <input type="text" class="form-input" placeholder="e.g. has_citations" style="font-size:0.82rem;padding:7px 10px;" data-field="name" />
      </div>
      <div style="flex:1;min-width:160px;">
        <label class="form-label" style="font-size:0.75rem;">Type</label>
        <select class="form-select" style="font-size:0.82rem;padding:7px 10px;" data-field="type" onchange="updateRuleExtras(this,'${id}')">
          ${typeOptions}
        </select>
      </div>
      <div style="width:80px;">
        <label class="form-label" style="font-size:0.75rem;">Weight</label>
        <input type="number" class="form-input" value="1.0" step="0.1" min="0" max="10" style="font-size:0.82rem;padding:7px 10px;" data-field="weight" />
      </div>
      <button onclick="document.getElementById('rule-${id}').remove()" style="background:none;border:none;color:var(--text3);cursor:pointer;padding:28px 4px 0;font-size:1rem;">✕</button>
    </div>
    <div id="rule-extras-${id}" style="margin-top:8px;"></div>`;
  container.appendChild(div);
}

function updateRuleExtras(select, id) {
  const type = select.value;
  const extrasEl = document.getElementById(`rule-extras-${id}`);
  extrasEl.innerHTML = '';
  if (type === 'keyword_match') {
    extrasEl.innerHTML = `<div><label class="form-label" style="font-size:0.75rem;">Keywords (comma-separated)</label><input type="text" class="form-input" placeholder="step-by-step, therefore, because" style="font-size:0.82rem;padding:7px 10px;" data-extra="keywords" /></div>`;
  } else if (type === 'length_range') {
    extrasEl.innerHTML = `<div style="display:flex;gap:10px;"><div style="flex:1;"><label class="form-label" style="font-size:0.75rem;">Min chars</label><input type="number" class="form-input" value="100" style="font-size:0.82rem;padding:7px 10px;" data-extra="min_length" /></div><div style="flex:1;"><label class="form-label" style="font-size:0.75rem;">Max chars</label><input type="number" class="form-input" value="2000" style="font-size:0.82rem;padding:7px 10px;" data-extra="max_length" /></div></div>`;
  } else if (type === 'format_check') {
    extrasEl.innerHTML = `<div><label class="form-label" style="font-size:0.75rem;">Format</label><select class="form-select" style="font-size:0.82rem;padding:7px 10px;" data-extra="format"><option value="json">JSON</option><option value="markdown">Markdown</option><option value="numbered_list">Numbered list</option></select></div>`;
  } else if (type === 'no_hallucination') {
    extrasEl.innerHTML = `<div><label class="form-label" style="font-size:0.75rem;">Forbidden phrases (comma-separated)</label><input type="text" class="form-input" placeholder="I don't know, I'm not sure" style="font-size:0.82rem;padding:7px 10px;" data-extra="forbidden_phrases" /></div>`;
  } else if (type === 'custom_regex') {
    extrasEl.innerHTML = `<div><label class="form-label" style="font-size:0.75rem;">Regex pattern</label><input type="text" class="form-input" placeholder="\\bsource:\\b|\\[\\d+\\]" style="font-size:0.82rem;padding:7px 10px;" data-extra="pattern" /></div>`;
  }
}

function collectRules() {
  const rules = [];
  document.querySelectorAll('#rulesContainer > [id^="rule-"]').forEach(div => {
    const rule = {};
    div.querySelectorAll('[data-field]').forEach(inp => {
      const k = inp.dataset.field;
      rule[k] = k === 'weight' ? parseFloat(inp.value) : inp.value;
    });
    div.querySelectorAll('[data-extra]').forEach(inp => {
      const k = inp.dataset.extra;
      const v = inp.value.trim();
      if (k === 'keywords' || k === 'forbidden_phrases') {
        rule[k] = v.split(',').map(s => s.trim()).filter(Boolean);
      } else if (k === 'min_length' || k === 'max_length') {
        rule[k] = parseInt(v, 10);
      } else {
        rule[k] = v;
      }
    });
    if (rule.name && rule.type) rules.push(rule);
  });
  return rules;
}

async function createRewardConfig() {
  const name = document.getElementById('newRewardName').value.trim();
  const algorithm = document.getElementById('newRewardAlgo').value;
  if (!name) { toast('Config name required', 'error'); return; }
  const rules = collectRules();
  if (rules.length === 0) { toast('Add at least one rule', 'error'); return; }
  try {
    await apiPost('/rewards', { name, algorithm, rules });
    toast('Reward config created', 'success');
    closeModal('createRewardModal');
    document.getElementById('newRewardName').value = '';
    document.getElementById('rulesContainer').innerHTML = '';
    loadRewards();
  } catch(e) { toast(e.message, 'error'); }
}

async function deleteRewardConfig(id) {
  if (!confirm('Delete this reward config and all computations?')) return;
  try {
    await apiDelete(`/rewards/${id}`);
    toast('Deleted', 'success');
    loadRewards();
  } catch(e) { toast(e.message, 'error'); }
}

// ── Modals ──
function openModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('open');
  if (id === 'createRewardModal') {
    loadRuleTypes();
    if (document.getElementById('rulesContainer').children.length === 0) addRule();
  }
}

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('open');
}

// Close modal on backdrop click
document.querySelectorAll('.modal-overlay').forEach(overlay => {
  overlay.addEventListener('click', e => {
    if (e.target === overlay) overlay.classList.remove('open');
  });
});

// ── Helpers ──
function formatBytes(bytes) {
  if (!bytes) return '0 B';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// ── Init ──
(async function init() {
  const t = getToken();
  if (!t) { window.location.href = '/'; return; }
  await loadUser();
  document.getElementById('loadingOverlay').style.display = 'none';
  document.getElementById('dashboardLayout').style.display = 'flex';
  loadOverview();
})();
