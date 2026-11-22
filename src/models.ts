import {
  BigAutoField,
  CharField,
  DateTimeField,
  JSONField,
  type Manager,
  Model,
  PositiveIntegerField,
  TextField,
} from '@ghuri/orm'

function emptyObject(): Record<string, unknown> {
  return {}
}

/**
 * The manifest: one row per indexed document (table `vectors_vectordocument`). It records how many chunks a document
 * has and a digest of what was embedded, so unchanged documents are not embedded again and stores without native
 * deletes (Vectorize) can delete a document's vectors.
 */
export class VectorDocument extends Model({
  id: new BigAutoField({ primaryKey: true }),
  /** The index name (`'blog.posts'`). */
  index: new CharField({ maxLength: 100 }),
  /** The store namespace (`'blog.posts|t=42'`), which carries the tenant. */
  namespace: new CharField({ maxLength: 255, blank: true, default: '' }),
  key: new CharField({ maxLength: 255 }),
  chunks: new PositiveIntegerField({ default: 0 }),
  /** SHA-256 of the text, metadata, embedding model and chunking. */
  digest: new CharField({ maxLength: 64 }),
  indexedAt: new DateTimeField({ autoNow: true }),
}) {
  declare static objects: Manager<VectorDocument>
  static Meta = {
    appLabel: 'vectors',
    ordering: ['index', 'namespace', 'key'],
    verboseName: 'vector document',
    uniqueTogether: [['index', 'namespace', 'key']],
  }

  toString(): string {
    return `${this.index}:${this.key}`
  }
}

/**
 * A vector of a `databaseStore()` (table `vectors_vectorembedding`): base64 float32 in a text column, so it works on
 * D1, SQLite and Postgres alike.
 */
export class VectorEmbedding extends Model({
  id: new BigAutoField({ primaryKey: true }),
  /** The `VECTORS` alias. */
  store: new CharField({ maxLength: 100 }),
  vectorId: new CharField({ maxLength: 64 }),
  namespace: new CharField({ maxLength: 255, blank: true, default: '' }),
  /** `metadata._key`: the document the vector belongs to. */
  document: new CharField({ maxLength: 255, blank: true, default: '' }),
  embedding: new TextField(),
  metadata: new JSONField({ default: emptyObject, blank: true }),
}) {
  declare static objects: Manager<VectorEmbedding>
  static Meta = {
    appLabel: 'vectors',
    verboseName: 'vector embedding',
    uniqueTogether: [['store', 'vectorId']],
    indexes: [{ fields: ['store', 'namespace', 'document'], name: 'vectors_embedding_doc_idx' }],
  }

  toString(): string {
    return `${this.store}:${this.vectorId}`
  }
}
