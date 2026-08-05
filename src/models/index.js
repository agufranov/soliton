import { kpI, kpII } from './kp.js';
import { zk } from './zk.js';
import { nlsCubic, nlsCQ, nlsSat } from './nls.js';
import { sineGordon } from './sineGordon.js';

// Ordered roughly "most soliton-like first".
export const MODELS = [kpI, nlsCQ, nlsSat, zk, kpII, nlsCubic, sineGordon];

export function modelById(id) {
  return MODELS.find((m) => m.id === id) || MODELS[0];
}
