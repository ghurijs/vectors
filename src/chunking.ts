/** How documents are split before embedding. Sizes are characters. */
export interface ChunkOptions {
  /** Maximum characters per chunk (default 1000). */
  size?: number
  /** Characters repeated from the end of a chunk at the start of the next (default 200). */
  overlap?: number
  /** Where to split, tried in order (default paragraphs, lines, sentences, words, then characters). */
  separators?: string[]
}

export const DEFAULT_CHUNK_SIZE = 1000
export const DEFAULT_CHUNK_OVERLAP = 200
export const DEFAULT_SEPARATORS = ['\n\n', '\n', '. ', ' ', '']

/** Split `text` into pieces no longer than `size`, preferring the earliest separator that works. */
function split(text: string, size: number, separators: string[]): string[] {
  if (text.length <= size) return [text]
  const idx = separators.findIndex((s) => s === '' || text.includes(s))
  const sep = idx < 0 ? '' : separators[idx]
  const rest = idx < 0 ? [] : separators.slice(idx + 1)
  if (sep === '') {
    const out: string[] = []
    for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size))
    return out
  }
  const out: string[] = []
  const parts = text.split(sep)
  for (let i = 0; i < parts.length; i++) {
    // Keep the separator with the piece it ends (so sentences keep their full stop).
    const piece = i < parts.length - 1 ? parts[i] + sep : parts[i]
    if (!piece) continue
    if (piece.length <= size) out.push(piece)
    else out.push(...split(piece, size, rest))
  }
  return out
}

/**
 * A recursive character splitter (LangChain's `RecursiveCharacterTextSplitter`): split on paragraphs, then lines,
 * sentences, words and characters until every piece fits, then merge pieces back into chunks of up to `size`
 * characters that overlap by about `overlap` characters. Whitespace-only chunks are dropped.
 */
export function chunkText(text: string, opts: ChunkOptions = {}): string[] {
  const size = opts.size ?? DEFAULT_CHUNK_SIZE
  const overlap = Math.min(opts.overlap ?? DEFAULT_CHUNK_OVERLAP, Math.floor(size / 2))
  if (!(size > 0)) throw new RangeError('chunk size must be > 0')
  if (overlap < 0) throw new RangeError('chunk overlap must be >= 0')
  const source = text.replace(/\r\n/g, '\n')
  if (!source.trim()) return []
  const pieces = split(source, size, opts.separators ?? DEFAULT_SEPARATORS)

  const chunks: string[] = []
  const current: string[] = []
  let length = 0
  const flush = () => {
    const chunk = current.join('').trim()
    if (chunk) chunks.push(chunk)
  }
  for (const piece of pieces) {
    if (length + piece.length > size && current.length) {
      flush()
      // Start the next chunk with the tail of this one, up to `overlap` characters, in whole pieces.
      while (current.length && (length > overlap || length + piece.length > size)) {
        length -= current[0].length
        current.shift()
      }
    }
    current.push(piece)
    length += piece.length
  }
  flush()
  return chunks
}
