import { GhuriError } from '@ghuri/core'

/** Base class of every `@ghuri/vectors` error. */
export class VectorsError extends GhuriError {}

/** A `VECTORS` entry is missing or is not a vector store. */
export class InvalidVectorStore extends VectorsError {}

/** An embedding model could not be resolved or its call failed. */
export class EmbeddingError extends VectorsError {}

/** A vector's length does not match the store's dimensions. */
export class DimensionMismatch extends VectorsError {}

/** A metadata filter uses an unknown lookup or a value the store cannot compare. */
export class FilterError extends VectorsError {}

/** A tenant-scoped index was used without a current tenant (it fails closed). */
export class TenantRequired extends VectorsError {}
