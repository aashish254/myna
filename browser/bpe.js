/** The trained tokenizer, in the browser, from `tokenizer.json` alone.
 *
 * `myna.tokenizer` builds a ByteLevel BPE with `tokenizers`, and the artifact's contract
 * is that a page must reproduce its ids exactly: a single token that differs shifts every
 * option span, and a wrong span means a wrong decision with nothing printed to say so.
 * So this file re-implements the three stages the Python object runs — added-token
 * matching, whitespace split, byte-level mapping, greedy BPE merge — rather than pulling
 * in a WASM tokenizer to do it opaquely.
 *
 * Two behaviours are load-bearing and easy to get wrong, and both come from the config
 * this repo trains with (`tokenizer.py`):
 *
 * * `pre_tokenizer` is `WhitespaceSplit` followed by ByteLevel with `use_regex: false`,
 *   so the separators are *deleted*, not turned into `Ġ`. "the quick" and "thequick"
 *   share a prefix, and no space symbol ever appears in a sequence.
 * * ByteLevel maps the string's UTF-8 **bytes**, so "é" arrives as two symbols. Printable
 *   bytes keep their own code point; the rest shift up into the U+0100 range.
 *
 * There is deliberately no word-length cutoff here, though `tokenizers` documents a
 * `max_word_length` of 64: this tokenizer writes no such key, and measured against it a
 * 260-character word still merges normally ("ab" x 130 -> 130 tokens, not 260 symbols).
 * Copying the documented default would have split every long word into single bytes and
 * moved every span after it — which is the failure this file exists to prevent, so the
 * claim is checked against the artifact rather than read off the library's docs.
 */

/** HF's ByteLevel byte -> symbol: 0x21-0x7e, 0xa1-0xac, 0xae-0xff map to themselves and
 *  everything else (control bytes, space, the C1 range, U+00AD) lands in 0x100+. */
export function byteToSymbol(b) {
  if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
    return String.fromCharCode(b);
  }
  return String.fromCharCode(0x100 + ByteLevelBPE.OTHER_BYTES.indexOf(b));
}

export class ByteLevelBPE {
  static OTHER_BYTES = (() => {
    const out = [];
    for (let b = 0; b < 256; b++) {
      if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))) {
        out.push(b);
      }
    }
    return out;
  })();

  constructor(tokenizerJson) {
    const model = tokenizerJson.model;
    if (model.type !== "BPE") {
      throw new Error(`tokenizer is ${model.type}, and only BPE is implemented here`);
    }
    this.vocab = new Map(Object.entries(model.vocab).map(([k, v]) => [k, v]));
    this.ranks = new Map();
    model.merges.forEach((pair, i) => {
      const key = Array.isArray(pair) ? pair.join(" ") : pair;
      if (!this.ranks.has(key)) this.ranks.set(key, i);
    });
    // Added tokens are matched before anything else sees the text, longest first.
    this.added = (tokenizerJson.added_tokens || [])
      .slice()
      .sort((a, b) => b.content.length - a.content.length)
      .map((t) => [t.content, t.id]);
    this.splitter = this.added.length
      ? new RegExp(`(${this.added.map(([c]) => escapeRe(c)).join("|")})`, "g")
      : null;
    this.padId = this.vocab.get("[PAD]");
    if (this.padId === undefined) throw new Error("tokenizer has no [PAD]: nothing to pad with");
  }

  static fromJSON(text) {
    return new ByteLevelBPE(typeof text === "string" ? JSON.parse(text) : text);
  }

  /** The ids of one piece of plain text — exactly `myna.tokenizer.encode_text`. */
  encode(text) {
    if (!this.splitter) return this.encodePlain(text);
    const ids = [];
    for (const piece of text.split(this.splitter)) {
      if (piece === "" || piece === undefined) continue;
      const special = this.added.find(([c]) => c === piece);
      if (special) ids.push(special[1]);
      else ids.push(...this.encodePlain(piece));
    }
    return ids;
  }

  encodePlain(text) {
    const ids = [];
    for (const word of text.split(/\s+/)) {
      if (!word) continue;
      for (const part of this.mergeWord(symbolsOf(word))) {
        const id = this.vocab.get(part);
        if (id === undefined) {
          // A silent drop here is the bug this whole file exists to prevent: the spans
          // after it would still be numbers, just the wrong ones.
          throw new Error(`tokenizer cannot represent "${part}" (U+${part.codePointAt(0)
            .toString(16)}): no vocabulary entry`);
        }
        ids.push(id);
      }
    }
    return ids;
  }

  /** Greedy merge of one word's symbols: lowest rank, leftmost on a tie, one occurrence
   *  at a time — which is what `tokenizers`' priority queue reduces to. */
  mergeWord(symbols) {
    const parts = symbols.slice();
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      parts.splice(best, 1, parts[best] + parts[best + 1]);
      parts.splice(best + 1, 1);
    }
    return parts;
  }

  idToToken(id) {
    if (!this._inv) this._inv = new Map([...this.vocab].map(([k, v]) => [v, k]));
    return this._inv.get(id);
  }

  decode(ids) {
    const out = [];
    for (const id of ids) {
      const tok = this.idToToken(id);
      if (tok === undefined || this.isSpecial(tok)) continue;
      for (const c of tok) {
        const code = c.charCodeAt(0);
        out.push(code >= 0x100 ? ByteLevelBPE.OTHER_BYTES[code - 0x100] : code);
      }
    }
    return new TextDecoder().decode(Uint8Array.from(out));
  }

  isSpecial(tok) {
    return this.added.some(([c]) => c === tok);
  }

  /** Mirror of `myna.tokenizer.build_question`: instruction, then `[OPT]` before each
   *  option's own text, then `[DECIDE]`; `spans[i]` is option i's token range. */
  buildQuestion(instruction, options) {
    const opt = this.vocab.get("[OPT]");
    const dec = this.vocab.get("[DECIDE]");
    if (opt === undefined || dec === undefined) {
      throw new Error("tokenizer is missing [OPT] or [DECIDE]: the layout cannot be built");
    }
    const ids = this.encode(instruction);
    const spans = [];
    for (const o of options) {
      ids.push(opt);
      const start = ids.length;
      ids.push(...this.encode(o));
      spans.push([start, ids.length]);
    }
    ids.push(dec);
    return { ids, spans, decide: ids.length - 1 };
  }
}

function symbolsOf(word) {
  return [...new TextEncoder().encode(word)].map(byteToSymbol);
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
