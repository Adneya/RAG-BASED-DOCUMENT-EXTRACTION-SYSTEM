const form = document.querySelector("#chatForm");
const queryInput = document.querySelector("#query");
const fileInput = document.querySelector("#files");
const fileList = document.querySelector("#fileList");
const messages = document.querySelector("#messages");
const statusBadge = document.querySelector("#statusBadge");
const extractedText = document.querySelector("#extractedText");
const planTrace = document.querySelector("#planTrace");
const costEstimate = document.querySelector("#costEstimate");
const sendButton = document.querySelector("#sendButton");

fileInput.addEventListener("change", () => {
  const names = Array.from(fileInput.files).map((file) => file.name);
  fileList.textContent = names.length ? names.join(", ") : "";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  const files = Array.from(fileInput.files);

  if (!query && files.length === 0) {
    addMessage("assistant", "What would you like me to help with?");
    return;
  }

  addMessage("user", query || `${files.length} file(s) uploaded`);
  setBusy(true);

  const payload = new FormData();
  payload.append("query", query);
  files.forEach((file) => payload.append("files", file));

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      body: payload,
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(body || `Request failed with ${response.status}`);
    }

    const data = await response.json();
    addMessage("assistant", data.final_output || "No output returned.");
    renderExtracted(data.extracted_texts || []);
    renderTrace(data.plan_trace || []);
    renderEstimate(data.cost_estimate);
    statusBadge.textContent = data.status || "Completed";
    if (data.status === "needs_clarification" && data.follow_up_question) {
      queryInput.value = "";
      queryInput.placeholder = data.follow_up_question;
      queryInput.focus();
    }
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`);
    statusBadge.textContent = "Error";
  } finally {
    setBusy(false);
  }
});

function addMessage(role, text) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  const pre = document.createElement("pre");
  pre.textContent = text;
  node.appendChild(pre);
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

function renderExtracted(items) {
  extractedText.innerHTML = "";
  if (!items.length) {
    extractedText.textContent = "No extracted text for this run.";
    return;
  }

  items.forEach((item) => {
    const details = document.createElement("details");
    details.open = true;
    const summary = document.createElement("summary");
    const confidence =
      item.ocr_confidence === null || item.ocr_confidence === undefined
        ? ""
        : ` | OCR ${Number(item.ocr_confidence).toFixed(2)}`;
    summary.textContent = `${item.filename} | ${item.source_type}${confidence}`;
    const body = document.createElement("pre");
    body.textContent = item.text || "No readable text was extracted.";
    details.appendChild(summary);
    details.appendChild(body);

    if (item.warnings && item.warnings.length) {
      const warnings = document.createElement("ul");
      warnings.className = "warnings";
      item.warnings.forEach((warning) => {
        const li = document.createElement("li");
        li.textContent = warning;
        warnings.appendChild(li);
      });
      details.appendChild(warnings);
    }

    extractedText.appendChild(details);
  });
}

function renderTrace(steps) {
  planTrace.innerHTML = "";
  if (!steps.length) {
    const li = document.createElement("li");
    li.textContent = "No plan trace for this run.";
    planTrace.appendChild(li);
    return;
  }

  steps.forEach((step) => {
    const li = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = `${step.tool}: ${step.action}`;
    const meta = document.createElement("span");
    meta.textContent = `${step.status}${step.detail ? ` - ${step.detail}` : ""}`;
    li.appendChild(title);
    li.appendChild(meta);
    planTrace.appendChild(li);
  });
}

function renderEstimate(estimate) {
  if (!estimate) {
    costEstimate.textContent = "No estimate returned.";
    return;
  }
  const usd =
    estimate.estimated_usd === null || estimate.estimated_usd === undefined
      ? "not configured"
      : `$${Number(estimate.estimated_usd).toFixed(6)}`;
  costEstimate.textContent = `Input tokens: ${estimate.estimated_input_tokens}. Output tokens: ${estimate.estimated_output_tokens}. USD: ${usd}. ${estimate.note}`;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  fileInput.disabled = isBusy;
  queryInput.disabled = isBusy;
  statusBadge.textContent = isBusy ? "Working" : "Ready";
}

