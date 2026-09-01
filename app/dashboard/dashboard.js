const form = document.querySelector('#settings-form');
const statusText = document.querySelector('#save-status');
const saveButton = document.querySelector('#save-button');
let saved = null;
let availableModels = null;
let modelDiscoveryFailed = false;

const fields = {
  agent_name: document.querySelector('#agent-name'),
  system_prompt: document.querySelector('#system-prompt'),
  model: document.querySelector('#model'),
  memory_enabled: document.querySelector('#memory-enabled'),
  max_history_turns: document.querySelector('#history-turns'),
  user_profile: document.querySelector('#user-profile'),
  context_memory: document.querySelector('#context-memory'),
  volume: document.querySelector('#volume'),
};

function setStatus(text, type = '') {
  statusText.textContent = text;
  statusText.className = type;
}

function updateMemoryState() {
  const on = fields.memory_enabled.checked;
  document.querySelector('#memory-card').classList.toggle('disabled', !on);
  document.querySelector('#history-field').style.opacity = on ? '1' : '.5';
  fields.max_history_turns.disabled = !on;
}

function updateVolume() {
  document.querySelector('#volume-output').textContent = fields.volume.value;
}

function addModelOption(value, text, { disabled = false, selected = false } = {}) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = text;
  option.disabled = disabled;
  option.selected = selected;
  fields.model.appendChild(option);
}

function renderModelSelect(savedModel) {
  fields.model.replaceChildren();
  if (availableModels === null) {
    const label = modelDiscoveryFailed
      ? `${savedModel} (unavailable)`
      : `Loading models… (${savedModel})`;
    addModelOption('', label, { disabled: true, selected: true });
    return;
  }

  const savedIsAvailable = availableModels.includes(savedModel);
  if (!savedIsAvailable) {
    addModelOption('', `${savedModel} (unavailable)`, { disabled: true, selected: true });
  }
  for (const model of availableModels) {
    addModelOption(model, model, { selected: model === savedModel });
  }
}

function fill(settings) {
  saved = { ...settings };
  for (const [key, input] of Object.entries(fields)) {
    if (key === 'model') continue;
    if (input.type === 'checkbox') input.checked = Boolean(settings[key]);
    else input.value = settings[key] ?? '';
  }
  renderModelSelect(settings.model);
  updateMemoryState();
  updateVolume();
}

function payload() {
  return {
    agent_name: fields.agent_name.value.trim(),
    system_prompt: fields.system_prompt.value.trim(),
    model: fields.model.value,
    memory_enabled: fields.memory_enabled.checked,
    max_history_turns: Number(fields.max_history_turns.value || 0),
    user_profile: fields.user_profile.value.trim(),
    context_memory: fields.context_memory.value.trim(),
    volume: Number(fields.volume.value),
  };
}

function setConnection(devices, details = []) {
  const pill = document.querySelector('#connection-pill');
  const name = document.querySelector('#device-name');
  const identifiers = document.querySelector('#device-identifiers');
  const mac = document.querySelector('#device-mac');
  const ip = document.querySelector('#device-ip');
  const description = document.querySelector('#device-description');
  if (devices.length) {
    const first = details[0] || { device_id: devices[0], ip_address: null };
    pill.className = 'status-pill online';
    pill.innerHTML = '<span></span>Connected';
    name.textContent = devices.length === 1 ? 'LinkDog connected' : `${devices.length} LinkDogs connected`;
    identifiers.hidden = false;
    mac.textContent = first.device_id || devices[0];
    ip.textContent = first.ip_address || 'Unavailable';
    description.textContent = devices.length === 1
      ? 'LinkDog is connected to this adapter.'
      : 'Showing the first device connected to this adapter.';
  } else {
    pill.className = 'status-pill offline';
    pill.innerHTML = '<span></span>Offline';
    name.textContent = 'No LinkDog connected';
    identifiers.hidden = true;
    mac.textContent = '—';
    ip.textContent = '—';
    description.textContent = 'The dashboard remains available while the robot is offline.';
  }
}

async function loadModels(savedModel) {
  try {
    const response = await fetch('/api/models');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    availableModels = data.models || [];
    modelDiscoveryFailed = false;
    renderModelSelect(savedModel);
    return data.stale ? 'Settings loaded · model catalog is stale' : 'Settings loaded';
  } catch (error) {
    availableModels = null;
    modelDiscoveryFailed = true;
    renderModelSelect(savedModel);
    throw new Error(`Model discovery failed: ${error.message}`);
  }
}

async function load() {
  try {
    const response = await fetch('/api/settings');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    fill(data.settings);
    setConnection(data.connected_devices || [], data.connected_device_details || []);
    try {
      setStatus(await loadModels(data.settings.model));
    } catch (error) {
      setStatus(error.message, 'error');
    }
  } catch (error) {
    setStatus(`Could not load settings: ${error.message}`, 'error');
  }
}

for (const tab of document.querySelectorAll('.tab')) {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(button => {
      const active = button === tab;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    document.querySelectorAll('.panel').forEach(panel => {
      const active = panel.dataset.panel === tab.dataset.tab;
      panel.classList.toggle('active', active);
      panel.hidden = !active;
    });
  });
}

fields.memory_enabled.addEventListener('change', updateMemoryState);
fields.volume.addEventListener('input', updateVolume);
document.querySelector('#cancel-button').addEventListener('click', () => {
  if (saved) fill(saved);
  setStatus('Changes reset');
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  saveButton.disabled = true;
  setStatus('Saving…');
  try {
    const response = await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    fill(data.settings);
    setConnection(data.connected_devices || [], data.connected_device_details || []);
    setStatus(
      data.restart_required ? 'Saved · reconnect LinkDog to apply model and role' : 'Saved',
      'success',
    );
  } catch (error) {
    setStatus(`Could not save: ${error.message}`, 'error');
  } finally {
    saveButton.disabled = false;
  }
});

load();
