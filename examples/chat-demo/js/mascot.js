// Original Pixel Cat mascot (28x22 sprite on 2D canvas, zero dependencies) and 16x16 Context Donut Ring.

const W = 28;
const H = 22;
const OX = 4;
const OY = 5;

const EARS_UP = [
  ".ff.........ff.",
  ".fpf.......fpf.",
  ".fpfffffffffpf.",
];

const EARS_TWITCH = [
  ".ff..........ff",
  ".fpf.......fpf.",
  ".fpfffffffffpf.",
];

const EARS_DROOP = [
  "...............",
  "ffp.........pff",
  ".fpfffffffffpf.",
];

const BODY_ROWS = [
  ".ffffdfdfdffff.",
  "fffffffffffffff",
  "fffffffffffffff",
  "fffffffffffffff",
  "fffffffpfffffff",
  ".fffffffffffff.",
  "..fffffffffff..",
  "..fffffffffff..",
  ".fffffffffffff.",
  ".fffffffffffff.",
  ".fffffffffffff.",
  "..fff.....fff..",
];

const TAILS = [
  // Curled gently right
  [[14, 12, "f"], [15, 12, "f"], [16, 11, "f"], [17, 10, "f"], [17, 9, "f"], [16, 8, "d"], [17, 8, "d"]],
  // Swayed upright
  [[14, 12, "f"], [15, 12, "f"], [16, 11, "f"], [16, 10, "f"], [16, 9, "f"], [15, 8, "d"], [16, 8, "d"]],
  // Swayed out right
  [[14, 12, "f"], [15, 12, "f"], [16, 11, "f"], [17, 11, "f"], [18, 10, "f"], [18, 9, "d"], [19, 8, "d"]],
];

const TAIL_LOW = [
  [14, 13, "f"], [15, 13, "f"], [16, 13, "f"], [17, 12, "d"], [18, 12, "d"],
];

const SIGN_Z = ["aaa", "..a", ".a.", "aaa"];

function getCssColor(name, fallback) {
  try {
    const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return val || fallback;
  } catch {
    return fallback;
  }
}

function buildPalette() {
  const accent = getCssColor("--accent", "#b9c7ff");
  return {
    o: "#0d1018", // 1px outline
    f: "#eef0f6", // ivory-silver fur
    s: "#c8cedc", // fur shading
    d: "#96a0b8", // tabby forehead stripes & tail tip
    h: "#ffffff", // eye glint & fur highlight
    p: "#e99ab0", // inner ears & nose
    e: "#141824", // eyes, mouth & outlines
    w: "#7c8499", // whiskers (readable on both Graphite & Fog)
    b: "#ff9bb5", // cheek blush
    a: accent,    // accent props (thinking dots, magnifier, z)
    A: "#7f8cc4", // magnifier handle
    t: "#8fc0ff", // tear drop
  };
}

export function renderMascotGrid({ mood = "idle", now = 0, lastTokenAt = -1e9, blink = false, lookX = 0, reduced = false } = {}) {
  const step = Math.floor(now / (reduced ? 900 : 380));
  const talking = now - lastTokenAt < 350 && !["thinking", "tool", "sleepy", "sad"].includes(mood);
  const syllable = talking && Math.floor(now / 110) % 2 === 1;

  let dy = 0;
  if (mood === "sleepy" || mood === "sad") dy = 1;
  if (!reduced && (mood === "happy" || mood === "wave")) {
    dy = -[0, 1, 1, 0][Math.floor(now / 210) % 4];
  } else if (!reduced && mood === "idle") {
    // Gentle breathing shift every ~2.4s
    dy = Math.floor(now / 1200) % 2 === 1 ? 1 : 0;
  }

  const g = Array.from({ length: H }, () => Array(W).fill(null));
  const putAbs = (x, y, c) => {
    if (x >= 0 && y >= 0 && x < W && y < H) g[y][x] = c;
  };
  const B = (x, y, c) => putAbs(OX + x, OY + dy + y, c);
  const px = (list, c = "e") => list.forEach(([x, y]) => B(x, y, c));

  // Ears: droop on sad/sorry, occasional twitch on idle
  const earTwitch = !reduced && mood === "idle" && now > 0 && (Math.floor(now / 450) % 13 === 7);
  const ears = (mood === "sad" || mood === "sorry")
    ? EARS_DROOP
    : (earTwitch ? EARS_TWITCH : EARS_UP);
  const template = [...ears, ...BODY_ROWS];

  // Fill body with right/bottom soft shading
  for (let y = 0; y < template.length; y++) {
    const row = template[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === ".") continue;
      if (ch === "f") {
        const rightEmpty = x + 1 >= row.length || row[x + 1] === ".";
        const belowEmpty = y + 1 >= template.length || template[y + 1][x] === ".";
        B(x, y, (rightEmpty || belowEmpty) && y >= 4 ? "s" : "f");
      } else {
        B(x, y, ch);
      }
    }
  }
  // Subtle forehead highlight & bright front paws
  B(4, 4, "h");
  for (const x of [2, 3, 4, 10, 11, 12]) B(x, 14, "f");

  // Tail (sways rhythmically unless sleepy/sad)
  const tailCoords = (mood === "sleepy" || mood === "sad")
    ? TAIL_LOW
    : TAILS[(reduced ? 0 : (mood === "happy" || mood === "talking" ? Math.floor(now / 220) : step)) % TAILS.length];
  for (const [tx, ty, tc] of tailCoords) B(tx, ty, tc);

  // Whiskers (placed cleanly outside the dark cheek outline)
  const drawWhiskers = () => {
    putAbs(OX - 2, OY + dy + 6, "w");
    putAbs(OX - 2, OY + dy + 8, "w");
    putAbs(OX + 16, OY + dy + 6, "w");
    putAbs(OX + 16, OY + dy + 8, "w");
  };

  // Eyes & expressions
  const eye = (x, y, shine = true) => {
    B(x, y, "e");
    B(x + 1, y, "e");
    B(x, y + 1, "e");
    B(x + 1, y + 1, "e");
    if (shine) B(x + (lookX > 0 ? 1 : 0), y, "h");
  };
  const eyes = (y = 5, shine = true, lx = lookX) => {
    eye(3 + lx, y, shine);
    eye(10 + lx, y, shine);
  };
  const closedEyes = (y = 6) => px([[3, y], [4, y], [10, y], [11, y]]);
  const happyEyes = () => px([[3, 6], [4, 5], [5, 6], [9, 6], [10, 5], [11, 6]]);
  const catMouth = () => px([[6, 8], [8, 8]]);
  const blush = () => px([[2, 7], [12, 7]], "b");

  switch (mood) {
    case "listening":
      if (blink) closedEyes(6);
      else eyes(6, true, lookX);
      catMouth();
      break;
    case "thinking":
      eyes(5, false, 1);
      px([[7, 8]]);
      for (let i = 0; i < 3; i++) {
        if (step % 4 > i) putAbs(OX + 16 + i * 2, OY + dy - 2 + (i === 1 ? -1 : 0), "a");
      }
      break;
    case "talking":
      if (blink) closedEyes(6);
      else eyes(5, true, 0);
      if (syllable) {
        px([[6, 8], [8, 8], [7, 9]]);
        B(7, 8, "p");
      } else {
        catMouth();
      }
      break;
    case "tool":
      if (blink) closedEyes(6);
      else eyes(5, false, 1);
      catMouth();
      // Magnifying glass next to cat
      px([[16, 2], [17, 1], [18, 1], [19, 2], [19, 3], [18, 4], [17, 4], [16, 3]], "a");
      if (step % 3 !== 0) B(17, 2, "h");
      px([[15, 5], [14, 6]], "A");
      break;
    case "wave":
      happyEyes();
      catMouth();
      blush();
      // Raised waving paw on the left (clear of the tail on the right)
      px(
        Math.floor(now / 180) % 2
          ? [[-1, 9], [-2, 8], [-2, 7], [-2, 6]]
          : [[-1, 9], [-2, 8], [-3, 7], [-3, 6]],
        "f",
      );
      break;
    case "happy":
      happyEyes();
      catMouth();
      blush();
      break;
    case "sorry":
      closedEyes(6);
      px([[6, 8], [7, 9], [8, 8]]);
      break;
    case "sad":
      closedEyes(6);
      px([[6, 9], [7, 8], [8, 9]]);
      B(3, 7 + (step % 3), "t");
      break;
    case "sleepy":
      closedEyes(6);
      px([[7, 8]]);
      if (step % 6 < 4) {
        const up = step % 6 > 1 ? -4 : -3;
        SIGN_Z.forEach((row, j) => {
          [...row].forEach((ch, i) => {
            if (ch !== ".") putAbs(OX + 17 + i, OY + dy + up + j, "a");
          });
        });
      }
      break;
    default:
      if (blink) closedEyes(6);
      else eyes(5, true, lookX);
      catMouth();
      break;
  }

  // Automatic 1px dark outline around solid cat cells (excluding floating props)
  const solid = (x, y) => g[y] && ["f", "s", "d", "h", "p", "b", "e"].includes(g[y][x]);
  const edges = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!g[y][x] && (solid(x - 1, y) || solid(x + 1, y) || solid(x, y - 1) || solid(x, y + 1))) {
        edges.push([x, y]);
      }
    }
  }
  for (const [x, y] of edges) g[y][x] = "o";

  // Crisp subtle whiskers at outer cheek edges
  drawWhiskers();

  return g;
}

function paintSprite(ctx, opts = {}) {
  const g = renderMascotGrid(opts);
  const pal = buildPalette();
  ctx.clearRect(0, 0, W, H);
  ctx.globalAlpha = opts.mood === "sleepy" || opts.mood === "sad" ? 0.86 : 1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (g[y][x]) {
        ctx.fillStyle = pal[g[y][x]];
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }
  ctx.globalAlpha = 1;
}

export function drawStillMascot(canvas) {
  if (!canvas) return;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  paintSprite(ctx, { mood: "idle", now: 0, reduced: true });
}

export function createPixelMascot(canvas) {
  if (!canvas) {
    return { set() {}, token() {}, getState: () => "idle" };
  }
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;

  let mood = "idle";
  let lastTokenAt = -1e9;
  let blinkAt = performance.now() + 2600;
  let glanceAt = performance.now() + 4500;
  let glance = 0;
  let pointerX = 0;
  let lastPaint = -1;

  window.addEventListener("pointermove", event => {
    if (!canvas.offsetParent) return;
    const rect = canvas.getBoundingClientRect();
    const dx = (event.clientX - (rect.left + rect.width / 2)) / 280;
    pointerX = Math.max(-1, Math.min(1, dx));
  }, { passive: true });

  function tick(now) {
    if (!canvas.isConnected) return;
    if (ctx && !document.hidden && canvas.offsetParent !== null && now - lastPaint >= 55) {
      lastPaint = now;
      if (now > blinkAt + 140) blinkAt = now + 2600 + Math.random() * 3400;
      const blink = now > blinkAt && ["idle", "listening", "talking", "tool"].includes(mood);
      if (now > glanceAt) {
        glance = glance ? 0 : (Math.random() < 0.5 ? -1 : 1);
        glanceAt = now + (glance ? 850 : 3800 + Math.random() * 3800);
      }
      const lookX = ["idle", "listening"].includes(mood) ? (glance || Math.round(pointerX)) : 0;
      paintSprite(ctx, { mood, now, lastTokenAt, blink, lookX, reduced });
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  return {
    set(next) {
      if (next && next !== mood) {
        mood = next;
        lastPaint = -1;
      }
    },
    token() {
      lastTokenAt = performance.now();
    },
    getState() {
      return mood;
    },
  };
}

export function drawContextRing(canvas, fraction, { over = false, warn = false } = {}) {
  if (!canvas) return;
  const size = 16;
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const px = Math.round(size * dpr);
  if (canvas.width !== px || canvas.height !== px) {
    canvas.width = px;
    canvas.height = px;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, size, size);

  const cx = size / 2;
  const cy = size / 2;
  const r = 6;
  const track = getCssColor("--line2", "rgba(255,255,255,0.14)");
  const fill = over
    ? getCssColor("--danger", "#ff8f80")
    : warn
      ? getCssColor("--warn", "#f2c27a")
      : getCssColor("--accent", "#b9c7ff");

  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = track;
  ctx.stroke();

  const clamped = Math.max(0, Math.min(1, Number(fraction) || 0));
  if (clamped > 0.005) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + clamped * Math.PI * 2);
    ctx.strokeStyle = fill;
    ctx.lineCap = "round";
    ctx.stroke();
  }
  ctx.restore();
}
