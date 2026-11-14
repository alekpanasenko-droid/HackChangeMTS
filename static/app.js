const $ = id => document.getElementById(id);
const state = {sessionId: sessionStorage.getItem('energytool_session'), busy: false, debug: false, messages: []};
const escapeHtml = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path, options = {}) {
  const response = await fetch('/api' + path, {headers: {'Content-Type':'application/json'}, ...options});
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Ошибка запроса');
  return data;
}
function dateLabel(value) { return new Date(value).toLocaleDateString('ru-RU', {day:'numeric',month:'short'}); }
function timeLabel(value) { return new Date(value).toLocaleTimeString('ru-RU', {hour:'2-digit',minute:'2-digit'}); }
async function newSession() {
  const session = await api('/sessions', {method:'POST'});
  state.sessionId = session.id;
  sessionStorage.setItem('energytool_session', session.id);
  state.messages = [];
  renderMessages();
  $('lead-download').classList.add('hidden');
  await loadSessions();
}
async function loadSessions() {
  const rows = await api('/sessions');
  $('sessions').innerHTML = rows.map(row => `<button class="session ${row.id === state.sessionId ? 'active' : ''}" data-id="${row.id}"><span class="icon">☏</span><span class="copy"><strong>${escapeHtml(row.title || 'Новый диалог')}</strong><small>${dateLabel(row.updated_at)}${row.has_lead ? ' · заявка' : ''}</small></span></button>`).join('');
  $('sessions').querySelectorAll('.session').forEach(button => button.onclick = () => openSession(button.dataset.id));
}
async function openSession(id) {
  const session = await api('/sessions/' + encodeURIComponent(id));
  state.sessionId = id;
  sessionStorage.setItem('energytool_session', id);
  state.messages = session.messages;
  renderMessages();
  $('lead-download').classList.toggle('hidden', !session.lead);
  if (session.lead) $('pdf-link').href = `/api/leads/${session.lead.id}/pdf`;
  $('manager-offer').classList.toggle('hidden', !!session.lead || state.messages.filter(m => m.role === 'user').length < 2);
  $('sidebar').classList.remove('open');
  await loadSessions();
}
function renderMessages() {
  $('welcome').classList.toggle('hidden', state.messages.length > 0);
  $('messages').innerHTML = state.messages.map((message, index) => {
    const user = message.role === 'user';
    return `<div class="message ${user ? 'user' : 'assistant'}">${user ? '' : '<span class="avatar">✳</span>'}<div class="bubble">${escapeHtml(message.content)}<div class="message-foot"><span>${message.created_at ? timeLabel(message.created_at) : ''}</span>${user ? '' : `<button class="inspect" data-index="${index}">⌁ Показать источники</button>`}</div></div></div>`;
  }).join('');
  $('messages').querySelectorAll('.inspect').forEach(button => button.onclick = () => inspect(Number(button.dataset.index)));
  $('conversation').scrollTop = $('conversation').scrollHeight;
  if (!state.messages.length) $('manager-offer').classList.add('hidden');
}
function inspect(index) {
  const message = state.messages[index];
  const sources = message.sources || [];
  const trace = message.trace || {};
  $('debug-body').innerHTML = `<div class="metric"><span>Время ответа</span><b>${message.response_ms == null ? '—' : (message.response_ms / 1000).toFixed(2) + ' с'}</b></div><div class="agent-trace"><h3>РАБОТА АГЕНТОВ</h3><p><b>Поисковый агент</b><span>${trace.search_ms == null ? '—' : trace.search_ms + ' мс'} · ${escapeHtml(trace.search_mode || '—')}</span></p><p><b>Оркестратор</b><span>${escapeHtml(trace.decision || '—')} · уверенность ${trace.confidence == null ? '—' : Math.round(trace.confidence * 100) + '%'}</span></p><p><b>Ответ модели</b><span>${trace.generation_ms == null ? '—' : trace.generation_ms + ' мс'}</span></p>${trace.reason ? `<small>${escapeHtml(trace.reason)}</small>` : ''}</div><h3>ИСПОЛЬЗОВАННЫЕ ДОКУМЕНТЫ · ${sources.length}</h3>${sources.length ? sources.map(source => `<div class="source-card"><strong>${escapeHtml(source.source)}</strong><small>${escapeHtml(source.heading || 'Документ')} · строки ${source.line_start ?? '—'}–${source.line_end ?? '—'} · ${escapeHtml(source.id)}</small><p>${escapeHtml(source.text)}</p></div>`).join('') : '<p class="muted">Для этого ответа модель не указала подтверждающих фрагментов.</p>'}`;
  state.debug = true;
  $('debug-panel').classList.remove('hidden');
  $('debug-toggle').setAttribute('aria-pressed', 'true');
}
function setBusy(value) {
  state.busy = value;
  $('send').disabled = value;
  $('message-input').disabled = value;
  $('status').textContent = value ? 'Готовлю ответ…' : 'Готов к диалогу';
}
async function send(text) {
  text = text.trim();
  if (!text || state.busy) return;
  if (!state.sessionId) await newSession();
  $('message-input').value = '';
  state.messages.push({role:'user',content:text,created_at:new Date().toISOString(),sources:[]});
  renderMessages();
  setBusy(true);
  try {
    const result = await api('/chat', {method:'POST',body:JSON.stringify({session_id:state.sessionId,message:text})});
    state.messages.push({role:'assistant',content:result.answer,created_at:new Date().toISOString(),response_ms:result.response_ms,sources:result.sources,trace:result.trace});
    renderMessages();
    $('manager-offer').classList.toggle('hidden', !result.offer_manager);
    await loadSessions();
  } catch (error) {
    state.messages.pop();
    renderMessages();
    $('status').textContent = error.message;
    alert(error.message);
  } finally { setBusy(false); $('message-input').focus(); }
}
$('send').onclick = () => send($('message-input').value);
$('message-input').addEventListener('keydown', event => {if (event.key === 'Enter' && !event.shiftKey) {event.preventDefault();send(event.target.value);}});
$('new-chat').onclick = newSession;
$('menu').onclick = () => $('sidebar').classList.toggle('open');
$('debug-toggle').onclick = () => {state.debug = !state.debug;$('debug-panel').classList.toggle('hidden', !state.debug);$('debug-toggle').setAttribute('aria-pressed', String(state.debug));};
$('debug-close').onclick = () => {state.debug = false;$('debug-panel').classList.add('hidden');$('debug-toggle').setAttribute('aria-pressed','false');};
document.querySelectorAll('[data-prompt]').forEach(button => button.onclick = () => send(button.dataset.prompt));
$('open-lead').onclick = () => $('lead-dialog').showModal();
$('close-lead').onclick = () => $('lead-dialog').close();
$('lead-form').onsubmit = async event => {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(event.target));
  payload.session_id = state.sessionId;
  try {
    const lead = await api('/leads', {method:'POST',body:JSON.stringify(payload)});
    $('lead-dialog').close();
    $('manager-offer').classList.add('hidden');
    await openSession(state.sessionId);
    await loadSessions();
  } catch (error) {$('lead-error').textContent = error.message;}
};
(async () => {
  try {
    const status = await api('/status');
    $('status').textContent = status.ready ? `Готов к диалогу · поиск: ${status.search}` :
      !status.knowledge_base ? 'Добавьте базу знаний из архива' :
      !status.system_prompt ? 'Добавьте локальный системный промпт' : 'Подключение модели настроим позже';
    if (state.sessionId) await openSession(state.sessionId); else await newSession();
  } catch (error) { $('status').textContent = error.message; }
})();
