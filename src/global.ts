/** One value per process under `Symbol.for(key)`, even if this package is loaded twice. */
export function globalSingleton<T>(key: string, make: () => T): T {
  const g = globalThis as Record<symbol, unknown>
  const sym = Symbol.for(key)
  if (!(sym in g)) g[sym] = make()
  return g[sym] as T
}
