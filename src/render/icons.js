// Иконки к кнопкам «Сценарии» и «Что порождает клик».
//
// Иконка сценария не рисуется руками: она строится из тех же `items`, которые
// применит пресет — точка там, где встанет солитон, размер по амплитуде,
// стрелка по скорости. Поэтому она физически не может разойтись с тем, что
// произойдёт по нажатию: добавили пресет — иконка появилась сама, и появилась
// правильная. Руками нарисованы только шесть иконок «что порождает клик»:
// там показывать надо форму профиля, а её из чисел не достанешь.
//
// Скорость берётся из модели, а не из `items`: у KP она вообще не свободна
// (3(λ²+μ²), −6λ), у ZK направление всегда +x, и только у НУШ в пресете лежит
// готовый вектор. Экранная ось y смотрит вниз — так же, как физическая, потому
// что рендер кладёт строку j прямо в строку картинки.
//
// Все координаты — в квадрате 24×24 (viewBox), цвета — классами из index.html.

const S = 24;
const f = (v) => Number(v.toFixed(2));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const svg = (body) => `<svg class="ic" viewBox="0 0 ${S} ${S}" aria-hidden="true">${body}</svg>`;
const frame = '<rect class="fr" x="1.25" y="1.25" width="21.5" height="21.5" rx="4"/>';

// Стрелка от края точки: короткая линия плюс треугольник. Маркеров нет
// намеренно — <marker> требует id, а id в повторяющихся иконках столкнулись бы.
function arrow(cx, cy, vx, vy, r, cls = 'mk') {
  const len = Math.hypot(vx, vy);
  if (len < 1e-9) return '';
  const ux = vx / len, uy = vy / len;
  const x0 = cx + ux * (r + 0.8), y0 = cy + uy * (r + 0.8);
  const x1 = x0 + ux * 5, y1 = y0 + uy * 5;      // конец наконечника
  const bx = x1 - ux * 2.5, by = y1 - uy * 2.5;  // основание наконечника
  const nx = -uy * 1.25, ny = ux * 1.25;
  return `<line class="ar" x1="${f(x0)}" y1="${f(y0)}" x2="${f(bx)}" y2="${f(by)}"/>`
    + `<polygon class="${cls}" points="${f(x1)},${f(y1)} ${f(bx + nx)},${f(by + ny)}`
    + ` ${f(bx - nx)},${f(by - ny)}"/>`;
}

const dot = (cx, cy, r, cls = 'mk') => `<circle class="${cls}" cx="${f(cx)}" cy="${f(cy)}" r="${f(r)}"/>`;

// Линейный солитон KP-II: профиль обращается в ноль на x = cx − tan·(y − cy),
// где tan = slopeIndex·Lx/Ly. Ящик на телефоне не квадратный, но иконка —
// квадрат, так что наклон берётся в единицах ящика.
function lineMark(it) {
  const tan = it.slopeIndex || 0;
  const w = clamp(1.2 + (it.kappa || 0.5) * 2.2, 1.2, 3.2);
  return `<line class="ln" style="stroke-width:${f(w)}" x1="${f(S / 2 + tan * S / 2)}" y1="0"`
    + ` x2="${f(S / 2 - tan * S / 2)}" y2="${S}"/>`;
}

// Ящик пресета кладётся не во весь квадрат, а с полями: солитон у самого края
// (x = 0.15 в паре пресетов) должен остаться точкой со стрелкой, а не парой
// пикселей, обрезанных рамкой.
const pos = (v) => 3 + (v ?? 0.5) * 18;

function itemMark(model, it) {
  const px = pos(it.x), py = pos(it.y);
  switch (model.family) {
    case 'kp': {
      if (it.kind === 'line') return lineMark(it);
      const mu = it.mu ?? 0.8, la = it.lambda ?? 0;
      const r = clamp(1.1 + 1.8 * mu, 1.4, 3.4);
      return dot(px, py, r) + arrow(px, py, 3 * (la * la + mu * mu), -6 * la, r);
    }
    case 'zk': {
      const r = clamp(1.1 + 0.55 * (it.c ?? 1.5), 1.4, 3.4);
      return dot(px, py, r) + arrow(px, py, 1, 0, r);
    }
    case 'nls': {
      const r = clamp(2.2 * (it.amp ?? 1), 1.6, 3.2);
      return dot(px, py, r) + arrow(px, py, it.vx || 0, it.vy || 0, r);
    }
    case 'sg': {
      // Кольцо рисуется настоящим радиусом: R0 задан в единицах ящика (L = 60),
      // и разница между пресетом «одно кольцо» (R0 = 9) и «двумя» (R0 = 6) на
      // иконке видна ровно такая, какая она есть.
      const r = clamp(((it.R0 ?? 8) / (model.grid.L || 60)) * 18, 2.2, 8.5);
      const cls = (it.sign ?? 1) < 0 ? 'ln2' : 'ln';
      return `<circle class="${cls}" fill="none" cx="${f(px)}" cy="${f(py)}" r="${f(r)}"/>`;
    }
    default:
      return dot(px, py, 2);
  }
}

/** Иконка сценария: мини-карта ящика с тем, что пресет в него положит. */
export function presetIcon(model, preset) {
  return svg(frame + (preset.items || []).map((it) => itemMark(model, it)).join(''));
}

// Формы профилей. Тут именно рисунок, а не данные: горб, прямой гребень,
// кольцо — это то, чем они отличаются на глаз, и никакого числа за этим нет.
const BUMP = 'M2 17.5 C 7.5 17.5 7 6.5 12 6.5 C 17 6.5 16.5 17.5 22 17.5';
const KINDS = {
  // KP-I / KP-II: локализованный горб. Baseline — чтобы читалось как профиль.
  lump: `<line class="fr" x1="2" y1="17.5" x2="22" y2="17.5"/><path class="ln" d="${BUMP}"/>`,
  // KP-II: гребень поперёк всего ящика, поэтому две параллельные линии и рамка
  line: `${frame}<line class="ln" x1="7" y1="1.5" x2="17" y2="22.5"/>`
    + '<line class="ln" style="opacity:.45;stroke-width:1.1" x1="11" y1="1.5" x2="21" y2="22.5"/>',
  // ZK: тот же горб, но у него всегда есть направление
  soliton: `<line class="fr" x1="2" y1="17.5" x2="18" y2="17.5"/>`
    + `<path class="ln" d="M2 17.5 C 6.5 17.5 6 7.5 10.5 7.5 C 15 7.5 14.5 17.5 18 17.5"/>`
    + arrow(15.5, 11.5, 1, 0, 1.5),
  // НУШ: «пуля» — яркое пятно с ореолом и свободным направлением
  bullet: '<circle class="ln" style="opacity:.4" fill="none" cx="10" cy="12" r="6.5"/>'
    + dot(10, 12, 3.4) + arrow(10, 12, 1, 0, 3.4),
  // sine-Gordon: кольцевой кинк, знак — цветом и знаком в середине
  ring: '<circle class="ln" fill="none" cx="12" cy="12" r="8"/>'
    + '<line class="ln" x1="12" y1="8.5" x2="12" y2="15.5"/>'
    + '<line class="ln" x1="8.5" y1="12" x2="15.5" y2="12"/>',
  antiring: '<circle class="ln2" fill="none" cx="12" cy="12" r="8"/>'
    + '<line class="ln2" x1="8.5" y1="12" x2="15.5" y2="12"/>',
};

/** Иконка к «что порождает клик»; неизвестный вид — без иконки, а не с «?». */
export function kindIcon(kindId) {
  return KINDS[kindId] ? svg(KINDS[kindId]) : '';
}
