(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const loginShell = $("#login-shell");
  const appShell = $("#app-shell");
  const loginForm = $("#login-form");
  const loginError = $("#login-error");
  const connectionPill = $("#connection-pill");
  const connectionLabel = $("#connection-label");
  const toastRegion = $("#toast-region");

  let csrfToken = "";
  let socket = null;
  let reconnectTimer = null;
  let currentSnapshot = null;

  const text = (selector, value) => {
    const element = $(selector);
    if (element) element.textContent = value;
  };

  const escapeHtml = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const userLabel = (userId) => (userId ? `用户 ${String(userId).slice(-6)}` : "暂无");

  const toast = (message, error = false) => {
    const node = document.createElement("div");
    node.className = `toast${error ? " error" : ""}`;
    node.textContent = message;
    toastRegion.append(node);
    window.setTimeout(() => node.remove(), 4200);
  };

  const api = async (path, options = {}) => {
    const headers = new Headers(options.headers || {});
    if (options.body) headers.set("Content-Type", "application/json");
    if (options.method && options.method !== "GET") {
      headers.set("X-Lingxu-CSRF", csrfToken);
    }
    const response = await fetch(path, {
      credentials: "same-origin",
      ...options,
      headers,
    });
    if (response.status === 401) {
      showLogin();
      throw new Error("会话已失效，请重新登录。");
    }
    if (!response.ok) {
      const detail = (await response.text()).trim();
      throw new Error(detail || `请求失败 (${response.status})`);
    }
    return response.json();
  };

  const setConnection = (status) => {
    connectionPill.classList.remove("connected", "disconnected");
    if (status === "connected") {
      connectionPill.classList.add("connected");
      connectionLabel.textContent = "实时连接";
    } else if (status === "disconnected") {
      connectionPill.classList.add("disconnected");
      connectionLabel.textContent = "正在重连";
    } else {
      connectionLabel.textContent = "正在连接";
    }
  };

  const showLogin = () => {
    if (socket) socket.close();
    socket = null;
    csrfToken = "";
    appShell.classList.add("is-hidden");
    loginShell.classList.remove("is-hidden");
    $("#login-token").value = "";
    $("#login-token").focus();
  };

  const showApp = (session) => {
    csrfToken = session.csrf_token;
    text("#guild-id", session.guild_id);
    loginShell.classList.add("is-hidden");
    appShell.classList.remove("is-hidden");
    connectWebSocket();
  };

  const formatClock = (remainingMs) => {
    const seconds = Math.max(0, Math.ceil(Number(remainingMs || 0) / 1000));
    const minutes = Math.floor(seconds / 60);
    return `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  };

  const formatTime = (iso) => {
    if (!iso) return "—";
    const value = new Date(iso);
    if (Number.isNaN(value.getTime())) return "—";
    return new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(value);
  };

  const renderState = (snapshot) => {
    text("#meeting-state", snapshot.state || "IDLE");
    text("#event-sequence", snapshot.sequence || 0);
    text("#session-id", snapshot.session_id || "NO ACTIVE SESSION");
    const agenda = snapshot.current_agenda;
    text("#current-agenda-index", agenda ? String(agenda.position).padStart(2, "0") : "—");
    text("#current-agenda-title", agenda ? agenda.title : "等待当前议题");
    text("#state-kicker", snapshot.active ? "会议状态" : "当前没有活动会议");
    $$("[data-control]").forEach((button) => {
      button.disabled = !snapshot.active;
    });
  };

  const renderTimer = (timers) => {
    const timer = timers?.[0];
    if (!timer) {
      text("#timer-value", "--:--");
      text("#timer-type", "NO TIMER");
      text("#timer-status", "等待会议计时器");
      $("#timer-progress").style.width = "0%";
      return;
    }
    text("#timer-value", formatClock(timer.remaining_ms));
    text("#timer-type", timer.type);
    text("#timer-status", timer.status === "PAUSED" ? "计时已暂停" : "计时进行中");
    const assumedWindow = Math.max(60_000, Number(timer.remaining_ms));
    const percent = Math.min(100, Math.max(4, (Number(timer.remaining_ms) / assumedWindow) * 100));
    $("#timer-progress").style.width = `${percent}%`;
  };

  const renderAgenda = (agenda) => {
    text("#agenda-count", `${agenda.length} 个议题`);
    const list = $("#agenda-list");
    if (!agenda.length) {
      list.innerHTML = '<p class="empty-state">暂无议程</p>';
      return;
    }
    list.innerHTML = agenda
      .map((item) => {
        const stateClass =
          item.status === "ACTIVE" ? " active" : item.status === "COMPLETED" ? " complete" : "";
        return `
          <article class="agenda-item${stateClass}">
            <span>${String(item.position).padStart(2, "0")} · ${escapeHtml(item.status)}</span>
            <h3>${escapeHtml(item.title)}</h3>
            <p>${item.duration_minutes === null ? "无固定时长" : `${item.duration_minutes} 分钟`}${
              item.requires_vote
                ? ` · 需投票（${item.vote_duration_seconds} 秒）`
                : ""
            }</p>
          </article>`;
      })
      .join("");
  };

  const renderPeople = (participants) => {
    const present = participants.filter((person) => person.present).length;
    text("#participant-count", `${present} 在线`);
    const list = $("#people-list");
    if (!participants.length) {
      list.innerHTML = '<p class="empty-state">暂无参与者</p>';
      return;
    }
    list.innerHTML = participants
      .map(
        (person) => `
          <div class="person-row">
            <span class="avatar">${escapeHtml(String(person.user_id).slice(-2))}</span>
            <div>
              <strong>${escapeHtml(userLabel(person.user_id))}</strong>
              <small>${person.present ? "在语音频道" : "已离开"}</small>
            </div>
            <div class="consent-set" aria-label="同意状态">
              <i class="consent asr${person.consent.realtime_asr ? " on" : ""}" title="实时识别"></i>
              <i class="consent recording${person.consent.recording ? " on" : ""}" title="录音"></i>
              <i class="consent minutes${person.consent.minutes ? " on" : ""}" title="纪要"></i>
            </div>
          </div>`,
      )
      .join("");
  };

  const renderSpeaker = (speaker) => {
    const queued = speaker?.queued_user_ids || [];
    text("#current-speaker", userLabel(speaker?.current_user_id));
    text("#queue-count", `${queued.length} 等待`);
    const list = $("#queue-list");
    list.innerHTML = queued.length
      ? queued.map((userId) => `<li>${escapeHtml(userLabel(userId))}</li>`).join("")
      : '<li class="empty-state">队列为空</li>';
  };

  const renderVote = (vote) => {
    if (!vote) {
      text("#vote-mode", "未开启");
      text("#vote-title", "当前没有进行中的投票");
      $("#vote-options").innerHTML =
        '<p class="empty-state">结果将在这里以汇总形式显示</p>';
      return;
    }
    text("#vote-mode", vote.anonymous ? "匿名 · 汇总" : "公开 · 汇总");
    text("#vote-title", vote.title);
    const total = vote.options.reduce((sum, option) => sum + Number(option.count || 0), 0);
    $("#vote-options").innerHTML = vote.options
      .map((option) => {
        const percent = total ? Math.round((Number(option.count) / total) * 100) : 0;
        return `
          <div class="vote-option">
            <div><span>${escapeHtml(option.label)}</span><strong>${option.count} · ${percent}%</strong></div>
            <span class="vote-bar"><i style="width:${percent}%"></i></span>
          </div>`;
      })
      .join("");
  };

  const renderUtterances = (utterances) => {
    const list = $("#utterance-list");
    if (!utterances.length) {
      list.innerHTML = '<p class="empty-state">尚无最终识别结果</p>';
      return;
    }
    list.innerHTML = utterances
      .map(
        (item) => `
          <div class="utterance">
            <span>${escapeHtml(userLabel(item.user_id))}<br>${escapeHtml(formatTime(item.created_at))}</span>
            <p>${escapeHtml(item.text)}</p>
          </div>`,
      )
      .join("");
  };

  const renderScenario = (scenarios) => {
    const scenario = scenarios?.[0];
    if (!scenario) {
      text("#scenario-status", "—");
      $("#scenario-summary").innerHTML = '<p class="empty-state">没有已发布场景</p>';
      return;
    }
    text("#scenario-status", scenario.status);
    $("#scenario-summary").innerHTML = `
      <h3>${escapeHtml(scenario.name)}</h3>
      <div class="scenario-meta">
        <div><span>Key</span><strong>${escapeHtml(scenario.key)}</strong></div>
        <div><span>Version</span><strong>v${scenario.version}</strong></div>
        <div><span>States</span><strong>${scenario.state_count}</strong></div>
        <div><span>Digest</span><strong>${escapeHtml(scenario.source_sha256.slice(0, 10))}…</strong></div>
      </div>`;
  };

  const renderSimulator = (simulator) => {
    if (!simulator) {
      text("#sim-state", "未启动");
      text("#sim-sequence", "SEQ 0");
      $("#sim-actions").innerHTML =
        '<p class="empty-state">重置后显示当前可用事件</p>';
      return;
    }
    text("#sim-state", simulator.state);
    text("#sim-sequence", `SEQ ${simulator.sequence}`);
    const actions = $("#sim-actions");
    if (!simulator.possible_events.length) {
      actions.innerHTML = `<p class="empty-state">${
        simulator.terminal ? "模拟已到达终态" : "当前没有可用事件"
      }</p>`;
      return;
    }
    actions.innerHTML = simulator.possible_events
      .map(
        (event) =>
          `<button type="button" data-sim-event="${escapeHtml(event)}">${escapeHtml(event)}</button>`,
      )
      .join("");
  };

  const renderEvents = (events) => {
    const list = $("#event-list");
    if (!events.length) {
      list.innerHTML = '<p class="empty-state">暂无事件</p>';
      return;
    }
    list.innerHTML = [...events]
      .reverse()
      .map(
        (event) => `
          <div class="event-row">
            <code>#${event.sequence}</code>
            <strong>${escapeHtml(event.type)}</strong>
            <span>${escapeHtml(event.state_before)} → ${escapeHtml(event.state_after)} · ${escapeHtml(
              event.source,
            )}</span>
            <time>${escapeHtml(formatTime(event.occurred_at))}</time>
          </div>`,
      )
      .join("");
  };

  const render = (snapshot) => {
    currentSnapshot = snapshot;
    renderState(snapshot);
    renderTimer(snapshot.timers || []);
    renderAgenda(snapshot.agenda || []);
    renderPeople(snapshot.participants || []);
    renderSpeaker(snapshot.speaker || {});
    renderVote(snapshot.vote);
    renderUtterances(snapshot.final_utterances || []);
    renderScenario(snapshot.scenario || []);
    renderSimulator(snapshot.simulator);
    renderEvents(snapshot.recent_events || []);
    text("#last-updated", `更新于 ${formatTime(snapshot.generated_at)}`);
  };

  const connectWebSocket = () => {
    if (socket) socket.close();
    window.clearTimeout(reconnectTimer);
    setConnection("connecting");
    const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${scheme}//${window.location.host}/ws`);
    socket.addEventListener("open", () => setConnection("connected"));
    socket.addEventListener("message", (event) => {
      try {
        render(JSON.parse(event.data));
      } catch {
        toast("收到无法解析的实时数据。", true);
      }
    });
    socket.addEventListener("close", (event) => {
      socket = null;
      if (event.code === 1008) {
        showLogin();
        return;
      }
      setConnection("disconnected");
      reconnectTimer = window.setTimeout(connectWebSocket, 1800);
    });
    socket.addEventListener("error", () => setConnection("disconnected"));
  };

  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    loginError.textContent = "";
    const button = loginForm.querySelector("button");
    button.disabled = true;
    try {
      const response = await fetch("/api/login", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: $("#login-token").value }),
      });
      if (!response.ok) throw new Error("令牌无效，或本地控制台尚未启用。");
      showApp(await response.json());
    } catch (error) {
      loginError.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });

  $("#logout-button").addEventListener("click", async () => {
    try {
      await api("/api/logout", { method: "POST" });
    } catch {
      // The local session is cleared client-side even if the server restarted.
    }
    showLogin();
  });

  $(".control-row").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-control]");
    if (!button || button.disabled) return;
    const action = button.dataset.control;
    if (action === "stop" && !window.confirm("确认结束当前会议？此操作会进入会后处理流程。")) {
      return;
    }
    button.disabled = true;
    try {
      const result = await api(`/api/control/${encodeURIComponent(action)}`, { method: "POST" });
      render(result.snapshot);
      toast(`已执行：${button.textContent.trim()}`);
    } catch (error) {
      toast(error.message, true);
    } finally {
      button.disabled = !currentSnapshot?.active;
    }
  });

  const ttsText = $("#tts-text");
  ttsText.addEventListener("input", () => text("#tts-count", `${ttsText.value.length} / 300`));
  $("#tts-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.target.querySelector("button");
    button.disabled = true;
    text("#tts-result", "正在生成并发送播报……");
    try {
      const result = await api("/api/tts", {
        method: "POST",
        body: JSON.stringify({ text: ttsText.value }),
      });
      text(
        "#tts-result",
        result.mode === "voice" ? "播报已在语音频道完成。" : "语音未连接，已发送文字提示。",
      );
      ttsText.value = "";
      text("#tts-count", "0 / 300");
    } catch (error) {
      text("#tts-result", error.message);
    } finally {
      button.disabled = false;
    }
  });

  $("#sim-reset").addEventListener("click", async () => {
    try {
      const result = await api("/api/simulator/reset", { method: "POST" });
      renderSimulator(result.simulator);
      toast("模拟器已从已发布场景重置。");
    } catch (error) {
      toast(error.message, true);
    }
  });

  $("#sim-actions").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-sim-event]");
    if (!button) return;
    button.disabled = true;
    try {
      const result = await api("/api/simulator/dispatch", {
        method: "POST",
        body: JSON.stringify({ event: button.dataset.simEvent }),
      });
      renderSimulator(result.simulator);
    } catch (error) {
      toast(error.message, true);
      button.disabled = false;
    }
  });

  const restore = async () => {
    try {
      const response = await fetch("/api/session", { credentials: "same-origin" });
      if (!response.ok) {
        showLogin();
        return;
      }
      showApp(await response.json());
    } catch {
      showLogin();
      loginError.textContent = "无法连接本地主持台，请确认 Bot 服务正在运行。";
    }
  };

  restore();
})();
