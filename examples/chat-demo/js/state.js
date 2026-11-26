import { defaultSampling } from "./tasks/sampling.js";

export const STORAGE_KEY = "kgl.demo.conversations.v1";
export const THEME_KEY = "kgl.demo.theme";
export const KEY_STORAGE = "kgl.demo.proxy-key";
export const SCHEMA_VERSION = 1;

export const PARTIAL_LABELS = {
  length: "Token limit",
  stopped: "Stopped",
  empty: "Empty reply",
  interrupted: "Interrupted",
};

export const state = {
  conversations: [],
  active: null,
  newDraft: "",
  newSystem: "",
  newSampling: defaultSampling(),
  newSources: [],
  customPrompts: [],
  vision: false,
  canSend: false,
  running: false,
  controller: undefined,
  sessionDeadline: undefined,
  contextLimit: undefined,
  capabilities: {},
  supportsThinking: false,
  supportsJsonSchema: false,
  taskMode: null,
  activeTagFilter: "",
  dateFilter: "all",
  taskOptions: {
    targetWords: "",
    audience: "general",
    goal: "polish",
    tone: "clear",
  },
};

export function chatTitle(chat) {
  return chat?.title || chat?.messages[0]?.content || "New conversation";
}

export function activeSampling() {
  return state.active ? (state.active.sampling || defaultSampling()) : state.newSampling;
}

export function activeSources(create = false) {
  if (state.active) {
    if (Array.isArray(state.active.sources)) return state.active.sources;
    if (create) return (state.active.sources = []);
    return [];
  }
  return state.newSources;
}
