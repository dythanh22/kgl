export const SAMPLING_MODES = {
  preset: { label: "Theo preset (0.6 / 0.95 / 20)", values: null },
  precise: { label: "Chính xác (0.2)", values: { temperature: 0.2, top_p: 0.9 } },
  creative: { label: "Sáng tạo (0.85)", values: { temperature: 0.85, top_p: 0.95 } },
  custom: { label: "Tùy chỉnh", values: null },
};

export function defaultSampling() {
  return {
    mode: "preset",
    temperature: "",
    top_p: "",
    presence_penalty: "",
    repeat_penalty: "",
  };
}

export function sanitizeSampling(raw) {
  const base = defaultSampling();
  if (!raw || typeof raw !== "object") return base;
  const mode = Object.hasOwn(SAMPLING_MODES, raw.mode) ? raw.mode : "preset";
  const pick = key => (raw[key] === undefined || raw[key] === null ? "" : String(raw[key]).trim());
  return {
    mode,
    temperature: pick("temperature"),
    top_p: pick("top_p"),
    presence_penalty: pick("presence_penalty"),
    repeat_penalty: pick("repeat_penalty"),
  };
}

function parseOptionalNumber(raw, name) {
  if (raw === "" || raw === undefined || raw === null) return undefined;
  const num = Number(raw);
  if (!Number.isFinite(num)) throw new Error(`${name} phải là một số hợp lệ.`);
  return num;
}

// Validate against src/kgl/protocol.py ranges and gate optional penalties by worker capabilities.
export function buildSamplingOverrides(sampling, capabilities = {}) {
  const cfg = sanitizeSampling(sampling);
  if (cfg.mode === "preset") {
    return { overrides: {}, summary: "Theo preset (worker default: 0.6 / 0.95 / 20)" };
  }

  let temp;
  let topP;
  let presence;
  let repeat;

  if (cfg.mode === "precise" || cfg.mode === "creative") {
    const preset = SAMPLING_MODES[cfg.mode].values;
    temp = preset.temperature;
    topP = preset.top_p;
  } else {
    temp = parseOptionalNumber(cfg.temperature, "temperature");
    topP = parseOptionalNumber(cfg.top_p, "top_p");
    presence = parseOptionalNumber(cfg.presence_penalty, "presence_penalty");
    repeat = parseOptionalNumber(cfg.repeat_penalty, "repeat_penalty");
  }

  const overrides = {};
  if (temp !== undefined) {
    if (temp < 0 || temp > 2) throw new Error("temperature phải nằm trong khoảng 0 đến 2.");
    overrides.temperature = temp;
  }
  if (topP !== undefined) {
    if (topP <= 0 || topP > 1) throw new Error("top_p phải lớn hơn 0 và nhỏ hơn hoặc bằng 1.");
    overrides.top_p = topP;
  }
  if (presence !== undefined && capabilities?.presence_penalty === true) {
    if (presence < -2 || presence > 2) throw new Error("presence_penalty phải nằm trong khoảng -2 đến 2.");
    overrides.presence_penalty = presence;
  }
  if (repeat !== undefined && capabilities?.repeat_penalty === true) {
    if (repeat <= 0 || repeat > 3.4e38) throw new Error("repeat_penalty phải là số dương hữu hạn.");
    overrides.repeat_penalty = repeat;
  }

  const parts = Object.entries(overrides).map(([k, v]) => `${k}=${v}`);
  const summary = parts.length
    ? `Thực gửi: ${parts.join(" · ")}`
    : "Theo preset (chưa đặt giá trị tùy chỉnh)";
  return { overrides, summary };
}
