const MAGIC = 0x46554256; // "VBUF"
const VERSION = 0x00050000; // 0.5.0

/** Explicit pre-v0.6 compatibility reader. It intentionally preserves legacy behavior. */
export class LegacyV05Instance {
	public mem: Uint8Array;
	public view: DataView;
	public alignment: number;

	constructor(buffer: Uint8Array) {
		this.mem = buffer;
		this.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

		if (this.mem.length < 16 || this.view.getUint32(0, true) !== MAGIC) {
			throw new Error("Invalid Magic header");
		}

		const aShift = this.mem[8]!;
		this.alignment = 1 << aShift;
	}

	public getCol(targetId: number): ArrayBufferView | null {
		let curr = 16; // Start after Global Header
		const end = this.mem.length;

		while (curr + 16 <= end) {
			const anchor = this.view.getBigUint64(curr, true);

			// Skip padding
			if (anchor === 0n) {
				curr += 8;
				continue;
			}

			const id = Number((anchor >> 16n) & 0xFFFFn);
			const plen = Number((anchor >> 32n) & 0xFFFFn);

			// Check overflow bit (Bit 9)
			const hasOverflow = (anchor & (1n << 9n)) !== 0n;
			let n: bigint;
			let headerSize: number;

			if (hasOverflow) {
				n = this.view.getBigUint64(curr + 8, true);
				headerSize = 16;
			} else {
				n = (anchor >> 48n) & 0xFFFFn;
				headerSize = 8;
			}

			const dataStart = (curr + headerSize + (this.alignment - 1)) & ~(this.alignment - 1);
			const byteLength = Number(n) * (plen / 8);

			if (id === targetId) {
				const slice = this.mem.subarray(dataStart, dataStart + byteLength);
				const sem = Number(anchor & 0xFn);
				// Map to correct TypedArray based on plen and sem
				switch (plen) {
					case 8: return new Uint8Array(slice.buffer, slice.byteOffset, slice.byteLength);
					case 16: return new Uint16Array(slice.buffer, slice.byteOffset, slice.byteLength / 2);
					case 32:
						return sem === 1
							? new Float32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4)
							: new Int32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4);
					case 64:
						return sem === 1
							? new Float64Array(slice.buffer, slice.byteOffset, slice.byteLength / 8)
							: new BigInt64Array(slice.buffer, slice.byteOffset, slice.byteLength / 8);
					default: return slice;
				}
			}

			// Advance to next block
			curr = dataStart + byteLength;
			curr = (curr + 7) & ~7; // Align to 8-byte boundary for next anchor
		}

		return null;
	}
}

/** @deprecated Use LegacyV05Instance only for explicit legacy-v0.5 compatibility. */
export const VBufInstance = LegacyV05Instance;

export type V06Semantic = 0 | 1 | 2 | 3;
export type V06Physical = 0 | 1;

export interface V06Block {
	readonly blockStart: number;
	readonly payloadStart: number;
	readonly payloadLength: number;
	readonly payloadEnd: number;
	readonly nextBlockStart: number;
	readonly keyId: number;
	readonly semantic: V06Semantic;
	readonly physical: V06Physical;
	readonly continuation: boolean;
	readonly bitWidth: number;
	readonly count: bigint;
	readonly payloadAlignment: bigint;
}

export class V06ValidationError extends Error {
	constructor(public readonly code: string, message: string) {
		super(message);
		this.name = "V06ValidationError";
	}
}

const U64_MAX = (1n << 64n) - 1n;
const checkedAdd = (a: bigint, b: bigint): bigint => {
	const value = a + b;
	if (a < 0n || b < 0n || value > U64_MAX) throw new V06ValidationError("arithmetic_overflow", "u64 addition overflow");
	return value;
};
const checkedMul = (a: bigint, b: bigint): bigint => {
	const value = a * b;
	if (a < 0n || b < 0n || value > U64_MAX) throw new V06ValidationError("arithmetic_overflow", "u64 multiplication overflow");
	return value;
};
const alignUp = (value: bigint, alignment: bigint): bigint => {
	if (alignment <= 0n || (alignment & (alignment - 1n)) !== 0n) throw new V06ValidationError("misaligned", "invalid alignment");
	return checkedAdd(value, alignment - 1n) & ~(alignment - 1n);
};

/** Fully validated canonical vBuf v0.6 reader. */
export class VBufV06 {
	readonly view: DataView;
	readonly baseShift: number;
	readonly baseStep: number;
	readonly indefinite: boolean;
	readonly headerSize: number;
	readonly dataRegionStart: number;
	readonly dataRegionSize: bigint;
	private readonly validatedBlocks: V06Block[] = [];

	get blocks(): ReadonlyArray<V06Block> { return this.validatedBlocks; }

	constructor(readonly mem: Uint8Array) {
		if (mem.byteLength < 24) throw new V06ValidationError("too_short", "v0.6 header is shorter than 24 bytes");
		this.view = new DataView(mem.buffer, mem.byteOffset, mem.byteLength);
		if (mem[0] !== 0x56 || mem[1] !== 0x42 || mem[2] !== 0x55 || mem[3] !== 0x46) throw new V06ValidationError("bad_magic", "invalid vBuf magic");
		if (this.view.getUint32(4, true) !== 0x00060000) throw new V06ValidationError("unsupported_version", "unsupported vBuf version");
		this.baseShift = mem[8]!;
		if (this.baseShift < 3 || this.baseShift > 8) throw new V06ValidationError("invalid_base_shift", "BaseShift is outside 3..=8");
		this.baseStep = 2 ** this.baseShift;
		const flags = mem[9]!;
		if ((flags & ~1) !== 0) throw new V06ValidationError("invalid_flags", "unknown global flags");
		this.indefinite = (flags & 1) !== 0;
		this.headerSize = this.view.getUint16(10, true);
		if (this.headerSize < 24 || this.headerSize % 8 !== 0) throw new V06ValidationError("invalid_header_size", "invalid HeaderSize");
		if (this.headerSize > mem.byteLength) throw new V06ValidationError("too_short", "truncated extended header");
		if (this.view.getUint32(12, true) !== 0) throw new V06ValidationError("invalid_reserved", "reserved global field is non-zero");
		const declaredSize = this.view.getBigUint64(16, true);
		if (this.indefinite && declaredSize !== 0n) throw new V06ValidationError("length_mismatch", "indefinite stream has non-zero size");

		let extensionOffset = 24n;
		const headerSizeBig = BigInt(this.headerSize);
		while (extensionOffset < headerSizeBig) {
			const offset = this.toIndex(extensionOffset);
			this.requireRange(extensionOffset, checkedAdd(extensionOffset, 8n), "truncated extension header");
			const extensionFlags = this.view.getUint16(offset + 2, true);
			const extensionSize = BigInt(this.view.getUint32(offset + 4, true));
			if ((extensionFlags & ~1) !== 0 || extensionSize < 8n || extensionSize % 8n !== 0n) throw new V06ValidationError("invalid_extension", "invalid extension framing");
			const extensionEnd = checkedAdd(extensionOffset, extensionSize);
			if (extensionEnd > headerSizeBig) throw new V06ValidationError("invalid_extension", "extension exceeds HeaderSize");
			if ((extensionFlags & 1) !== 0) throw new V06ValidationError("unknown_required_extension", "unknown required extension");
			extensionOffset = extensionEnd;
		}

		const baseStepBig = BigInt(this.baseStep);
		const dataStart = alignUp(headerSizeBig, baseStepBig);
		this.requireRange(headerSizeBig, dataStart, "global padding is outside input");
		this.requireZero(headerSizeBig, dataStart);
		this.dataRegionStart = this.toIndex(dataStart);
		const physicalEnd = BigInt(mem.byteLength);
		const dataEnd = this.indefinite ? physicalEnd : checkedAdd(dataStart, declaredSize);
		if (!this.indefinite && dataEnd !== physicalEnd) throw new V06ValidationError("length_mismatch", "known DataRegionSize differs from input");
		this.dataRegionSize = dataEnd - dataStart;

		let blockStart = dataStart;
		let requiredContinuationKey: number | null = null;
		while (blockStart < dataEnd) {
			if (blockStart % baseStepBig !== 0n || (blockStart - dataStart) % baseStepBig !== 0n) throw new V06ValidationError("misaligned", "unaligned block start");
			this.requireRange(blockStart, checkedAdd(blockStart, 8n), "truncated block anchor");
			const anchor = this.view.getBigUint64(this.toIndex(blockStart), true);
			if (anchor === 0n) throw new V06ValidationError("invalid_anchor", "zero block anchor");
			const semantic = Number(anchor & 0xfn);
			const physical = Number((anchor >> 4n) & 0xfn);
			const continuation = (anchor & (1n << 8n)) !== 0n;
			const count64 = (anchor & (1n << 9n)) !== 0n;
			const payloadShift = Number((anchor >> 10n) & 0x3fn);
			const keyId = Number((anchor >> 16n) & 0xffffn);
			const bitWidth = Number((anchor >> 32n) & 0xffffn);
			const inlineCount = (anchor >> 48n) & 0xffffn;
			if (requiredContinuationKey !== null && requiredContinuationKey !== keyId) throw new V06ValidationError("invalid_continuation", "continued block does not match preceding KeyID");
			requiredContinuationKey = null;

			let count: bigint;
			let blockHeaderSize: bigint;
			if (count64) {
				this.requireRange(blockStart, checkedAdd(blockStart, 16n), "truncated extended count");
				if (inlineCount !== 0n) throw new V06ValidationError("noncanonical_count", "extended count has inline value");
				count = this.view.getBigUint64(this.toIndex(blockStart + 8n), true);
				if (count <= 65535n) throw new V06ValidationError("noncanonical_count", "small count uses extended form");
				blockHeaderSize = 16n;
			} else {
				count = inlineCount;
				blockHeaderSize = 8n;
			}
			this.validateRepresentation(semantic, physical, bitWidth, count);
			const combinedShift = this.baseShift + payloadShift;
			if (combinedShift > 63) throw new V06ValidationError("misaligned", "payload shift exceeds 63");
			const payloadAlignment = 1n << BigInt(combinedShift);
			const headerEnd = checkedAdd(blockStart, blockHeaderSize);
			const payloadStart = alignUp(headerEnd, payloadAlignment);
			this.requireRange(headerEnd, payloadStart, "payload padding is outside input");
			this.requireZero(headerEnd, payloadStart);
			const payloadBits = checkedMul(count, BigInt(bitWidth));
			const payloadLength = payloadBits / 8n + (payloadBits % 8n === 0n ? 0n : 1n);
			const payloadEnd = checkedAdd(payloadStart, payloadLength);
			if (payloadEnd > dataEnd) throw new V06ValidationError("truncated_block", "declared payload exceeds data region");
			const nextBlockStart = alignUp(payloadEnd, baseStepBig);
			this.validatedBlocks.push(Object.freeze({
				blockStart: this.toIndex(blockStart), payloadStart: this.toIndex(payloadStart),
				payloadLength: this.toIndex(payloadLength), payloadEnd: this.toIndex(payloadEnd),
				nextBlockStart: this.toSafeNumber(nextBlockStart), keyId, semantic: semantic as V06Semantic,
				physical: physical as V06Physical, continuation, bitWidth, count, payloadAlignment,
			}));
			if (continuation && (payloadEnd === dataEnd || nextBlockStart >= dataEnd)) throw new V06ValidationError("invalid_continuation", "final physical block cannot continue");
			requiredContinuationKey = continuation ? keyId : null;
			if (payloadEnd === dataEnd) break;
			if (nextBlockStart >= dataEnd) throw new V06ValidationError("length_mismatch", "final tail padding is not canonical");
			this.requireZero(payloadEnd, nextBlockStart);
			blockStart = nextBlockStart;
		}
		Object.freeze(this.validatedBlocks);
	}

	block(keyId: number, occurrence = 0): V06Block | undefined {
		return this.validatedBlocks.filter((block) => block.keyId === keyId)[occurrence];
	}

	getU8(keyId: number, occurrence = 0): Uint8Array { return this.typed(keyId, occurrence, 0, 8, Uint8Array); }
	getU16(keyId: number, occurrence = 0): Uint16Array { return this.typed(keyId, occurrence, 0, 16, Uint16Array); }
	getU32(keyId: number, occurrence = 0): Uint32Array { return this.typed(keyId, occurrence, 0, 32, Uint32Array); }
	getU64(keyId: number, occurrence = 0): BigUint64Array { return this.typed(keyId, occurrence, 0, 64, BigUint64Array); }
	getI8(keyId: number, occurrence = 0): Int8Array { return this.typed(keyId, occurrence, 2, 8, Int8Array); }
	getI16(keyId: number, occurrence = 0): Int16Array { return this.typed(keyId, occurrence, 2, 16, Int16Array); }
	getI32(keyId: number, occurrence = 0): Int32Array { return this.typed(keyId, occurrence, 2, 32, Int32Array); }
	getI64(keyId: number, occurrence = 0): BigInt64Array { return this.typed(keyId, occurrence, 2, 64, BigInt64Array); }
	getF32(keyId: number, occurrence = 0): Float32Array { return this.typed(keyId, occurrence, 1, 32, Float32Array); }
	getF64(keyId: number, occurrence = 0): Float64Array { return this.typed(keyId, occurrence, 1, 64, Float64Array); }
	getOpaque(keyId: number, occurrence = 0): Uint8Array { return this.typed(keyId, occurrence, 3, 8, Uint8Array); }

	private typed<T extends ArrayBufferView>(keyId: number, occurrence: number, semantic: number, width: number, ctor: { new(buffer: ArrayBufferLike, byteOffset: number, length: number): T }): T {
		const block = this.block(keyId, occurrence);
		if (!block) throw new V06ValidationError("not_found", "KeyID occurrence not found");
		if (block.semantic !== semantic || block.bitWidth !== width) throw new V06ValidationError("type_mismatch", "wire representation does not match requested type");
		const elementBytes = width / 8;
		const count = Number(block.count);
		const expectedLength = count * elementBytes;
		if (!Number.isSafeInteger(count) || !Number.isSafeInteger(expectedLength) || expectedLength !== block.payloadLength) throw new V06ValidationError("length_mismatch", "typed length differs from validated payload");
		const absoluteOffset = this.mem.byteOffset + block.payloadStart;
		if (absoluteOffset % elementBytes !== 0) throw new V06ValidationError("misaligned", "typed-array alignment is not satisfied");
		if (block.payloadStart > block.payloadEnd || block.payloadEnd > this.mem.byteLength) throw new V06ValidationError("length_mismatch", "validated payload unavailable");
		// Construction occurs only after exact range, length, representation, and alignment checks.
		return new ctor(this.mem.buffer, absoluteOffset, count);
	}

	private validateRepresentation(semantic: number, physical: number, width: number, count: bigint): void {
		const semanticValid =
			(semantic === 0 && [8, 16, 32, 64].includes(width)) ||
			(semantic === 1 && [32, 64].includes(width)) ||
			(semantic === 2 && [8, 16, 32, 64].includes(width)) ||
			(semantic === 3 && width === 8);
		if (!semanticValid) throw new V06ValidationError("invalid_representation", "invalid semantic/width combination");
		if (!((physical === 0 && count === 1n) || physical === 1)) throw new V06ValidationError("invalid_representation", "invalid physical/count combination");
	}

	private requireRange(start: bigint, end: bigint, message: string): void {
		if (start > end || end > BigInt(this.mem.byteLength)) throw new V06ValidationError("too_short", message);
	}
	private requireZero(start: bigint, end: bigint): void {
		this.requireRange(start, end, "padding outside input");
		for (let index = this.toIndex(start); index < this.toIndex(end); index++) {
			if (this.mem[index] !== 0) throw new V06ValidationError("nonzero_padding", "non-zero canonical padding");
		}
	}
	private toIndex(value: bigint): number {
		if (value < 0n || value > BigInt(this.mem.byteLength)) throw new V06ValidationError("too_short", "offset lies outside input");
		return this.toSafeNumber(value);
	}
	private toSafeNumber(value: bigint): number {
		const number = Number(value);
		if (value < 0n || !Number.isSafeInteger(number)) throw new V06ValidationError("arithmetic_overflow", "offset exceeds JavaScript exact integer range");
		return number;
	}
}

export interface V06BlockOptions {
	keyId: number;
	physical?: V06Physical;
	continuation?: boolean;
	payloadShift?: number;
}

/** Canonical portable vBuf v0.6 writer. BaseShift is always explicit. */
export class VBufV06Writer {
	private readonly chunks: Uint8Array[] = [];
	private readonly baseStep: number;
	private readonly dataRegionStart: number;
	private requiredContinuationKey: number | null = null;
	private blockCount = 0;

	constructor(readonly baseShift: number, readonly indefinite = false) {
		if (!Number.isInteger(baseShift) || baseShift < 3 || baseShift > 8) throw new V06ValidationError("invalid_base_shift", "BaseShift is outside 3..=8");
		this.baseStep = 2 ** baseShift;
		this.dataRegionStart = Number(alignUp(24n, BigInt(this.baseStep)));
		const header = new Uint8Array(this.dataRegionStart);
		const view = new DataView(header.buffer);
		header.set([0x56, 0x42, 0x55, 0x46]);
		view.setUint32(4, 0x00060000, true);
		header[8] = baseShift;
		header[9] = indefinite ? 1 : 0;
		view.setUint16(10, 24, true);
		this.chunks.push(header);
	}

	writeU8(options: V06BlockOptions, values: Uint8Array): void { this.writeBlock(options, 0, 8, BigInt(values.length), values); }
	writeU16(options: V06BlockOptions, values: Uint16Array): void { this.writeBlock(options, 0, 16, BigInt(values.length), this.encode(values.length, 2, (view, offset, index) => view.setUint16(offset, values[index]!, true))); }
	writeU32(options: V06BlockOptions, values: Uint32Array): void { this.writeBlock(options, 0, 32, BigInt(values.length), this.encode(values.length, 4, (view, offset, index) => view.setUint32(offset, values[index]!, true))); }
	writeU64(options: V06BlockOptions, values: BigUint64Array): void { this.writeBlock(options, 0, 64, BigInt(values.length), this.encode(values.length, 8, (view, offset, index) => view.setBigUint64(offset, values[index]!, true))); }
	writeI8(options: V06BlockOptions, values: Int8Array): void { this.writeBlock(options, 2, 8, BigInt(values.length), this.encode(values.length, 1, (view, offset, index) => view.setInt8(offset, values[index]!))); }
	writeI16(options: V06BlockOptions, values: Int16Array): void { this.writeBlock(options, 2, 16, BigInt(values.length), this.encode(values.length, 2, (view, offset, index) => view.setInt16(offset, values[index]!, true))); }
	writeI32(options: V06BlockOptions, values: Int32Array): void { this.writeBlock(options, 2, 32, BigInt(values.length), this.encode(values.length, 4, (view, offset, index) => view.setInt32(offset, values[index]!, true))); }
	writeI64(options: V06BlockOptions, values: BigInt64Array): void { this.writeBlock(options, 2, 64, BigInt(values.length), this.encode(values.length, 8, (view, offset, index) => view.setBigInt64(offset, values[index]!, true))); }
	writeF32(options: V06BlockOptions, values: Float32Array): void { this.writeBlock(options, 1, 32, BigInt(values.length), this.encode(values.length, 4, (view, offset, index) => view.setFloat32(offset, values[index]!, true))); }
	writeF64(options: V06BlockOptions, values: Float64Array): void { this.writeBlock(options, 1, 64, BigInt(values.length), this.encode(values.length, 8, (view, offset, index) => view.setFloat64(offset, values[index]!, true))); }
	writeOpaque(options: V06BlockOptions, values: Uint8Array): void { this.writeBlock(options, 3, 8, BigInt(values.length), values); }

	finish(): Uint8Array {
		if (this.requiredContinuationKey !== null) throw new V06ValidationError("unterminated_continuation", "final block has Continuation set");
		const output = new Uint8Array(this.offset());
		let offset = 0;
		for (const chunk of this.chunks) { output.set(chunk, offset); offset += chunk.length; }
		if (!this.indefinite) new DataView(output.buffer).setBigUint64(16, BigInt(output.length - this.dataRegionStart), true);
		return output;
	}

	private writeBlock(options: V06BlockOptions, semantic: V06Semantic, bitWidth: number, count: bigint, payload: Uint8Array): void {
		const physical = options.physical ?? 1;
		const continuation = options.continuation ?? false;
		const payloadShift = options.payloadShift ?? 0;
		if (!Number.isInteger(options.keyId) || options.keyId < 0 || options.keyId > 0xffff) throw new V06ValidationError("invalid_key", "KeyID is outside u16");
		if (this.requiredContinuationKey !== null && this.requiredContinuationKey !== options.keyId) throw new V06ValidationError("continuation_mismatch", "continued block has a different KeyID");
		if (!Number.isInteger(payloadShift) || payloadShift < 0 || this.baseShift + payloadShift > 63) throw new V06ValidationError("invalid_payload_shift", "payload shift exceeds 63");
		this.validateRepresentation(semantic, physical, bitWidth, count);
		const payloadBits = checkedMul(count, BigInt(bitWidth));
		const expected = payloadBits / 8n + (payloadBits % 8n === 0n ? 0n : 1n);
		if (expected !== BigInt(payload.byteLength)) throw new V06ValidationError("payload_length_mismatch", "payload length differs from count and width");
		const payloadAlignment = 2 ** (this.baseShift + payloadShift);
		if (!Number.isSafeInteger(payloadAlignment)) throw new V06ValidationError("arithmetic_overflow", "payload alignment exceeds JavaScript exact integer range");

		if (this.blockCount > 0) this.padTo(this.baseStep);
		const extended = count > 65535n;
		const header = new Uint8Array(extended ? 16 : 8);
		const view = new DataView(header.buffer);
		let anchor = BigInt(semantic) | (BigInt(physical) << 4n);
		anchor |= BigInt(continuation ? 1 : 0) << 8n;
		anchor |= BigInt(extended ? 1 : 0) << 9n;
		anchor |= BigInt(payloadShift) << 10n;
		anchor |= BigInt(options.keyId) << 16n;
		anchor |= BigInt(bitWidth) << 32n;
		anchor |= (extended ? 0n : count) << 48n;
		view.setBigUint64(0, anchor, true);
		if (extended) view.setBigUint64(8, count, true);
		this.chunks.push(header);
		this.padTo(payloadAlignment);
		this.chunks.push(new Uint8Array(payload));
		this.blockCount++;
		this.requiredContinuationKey = continuation ? options.keyId : null;
	}

	private validateRepresentation(semantic: V06Semantic, physical: V06Physical, width: number, count: bigint): void {
		const semanticValid =
			((semantic === 0 || semantic === 2) && [8, 16, 32, 64].includes(width)) ||
			(semantic === 1 && [32, 64].includes(width)) || (semantic === 3 && width === 8);
		if (!semanticValid || !((physical === 0 && count === 1n) || physical === 1)) throw new V06ValidationError("invalid_representation", "invalid semantic/physical/count/width combination");
	}
	private encode(length: number, bytes: number, put: (view: DataView, offset: number, index: number) => void): Uint8Array {
		const byteLength = length * bytes;
		if (!Number.isSafeInteger(byteLength)) throw new V06ValidationError("arithmetic_overflow", "encoded byte length exceeds JavaScript exact integer range");
		const output = new Uint8Array(byteLength);
		const view = new DataView(output.buffer);
		for (let index = 0; index < length; index++) put(view, index * bytes, index);
		return output;
	}
	private offset(): number {
		const offset = this.chunks.reduce((total, chunk) => total + chunk.length, 0);
		if (!Number.isSafeInteger(offset)) throw new V06ValidationError("arithmetic_overflow", "writer offset exceeds JavaScript exact integer range");
		return offset;
	}
	private padTo(alignment: number): void {
		const padding = (alignment - (this.offset() % alignment)) % alignment;
		if (padding) this.chunks.push(new Uint8Array(padding));
	}
}

/** Explicit pre-v0.6 compatibility writer. */
export class LegacyV05Writer {
	private chunks: Uint8Array[] = [];
	private alignment: number;

	constructor(alignment: number = 4096) {
		this.alignment = alignment;
		const header = new Uint8Array(16);
		const view = new DataView(header.buffer);
		view.setUint32(0, MAGIC, true);
		view.setUint32(4, VERSION, true);
		header[8] = Math.log2(alignment);
		this.chunks.push(header);
	}

	private getOffset(): number {
		return this.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
	}

	public writeColumn(id: number, data: ArrayBufferView, bitWidth: number = 32) {
		const byteLength = data.byteLength;
		const n = BigInt(byteLength / (bitWidth / 8));

		const isFloat = data instanceof Float32Array || data instanceof Float64Array;
		const sem = isFloat ? 1n : 0n;

		// 1. The 64-bit Anchor
		let anchor = 0n;
		anchor |= sem;         // SEM: Float or Int
		anchor |= (1n << 4n);  // PHYS: Array
		anchor |= (1n << 9n);  // OVERFLOW: N follows as u64
		anchor |= (BigInt(id & 0xFFFF) << 16n);
		anchor |= (BigInt(bitWidth & 0xFFFF) << 32n);

		const header = new Uint8Array(16);
		const view = new DataView(header.buffer);
		view.setBigUint64(0, anchor, true);
		view.setBigUint64(8, n, true);
		this.chunks.push(header);

		// 2. Alignment to Diamond Grid
		const currentOffset = this.getOffset();
		const pad = (this.alignment - (currentOffset % this.alignment)) % this.alignment;
		if (pad > 0) {
			this.chunks.push(new Uint8Array(pad));
		}

		// 3. Data payload
		const rawBytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		this.chunks.push(rawBytes);

		// 4. Tail padding to 8-byte boundary for next anchor
		const endOffset = this.getOffset();
		const tailPad = (8 - (endOffset % 8)) % 8;
		if (tailPad > 0) {
			this.chunks.push(new Uint8Array(tailPad));
		}
	}

	public finish(): Uint8Array {
		const totalSize = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
		const output = new Uint8Array(totalSize);
		let offset = 0;
		for (const chunk of this.chunks) {
			output.set(chunk, offset);
			offset += chunk.length;
		}
		// Write final file size in Global Header
		const view = new DataView(output.buffer);
		view.setUint32(12, totalSize, true);
		return output;
	}
}

/** @deprecated Use LegacyV05Writer only for explicit legacy compatibility. */
export const VBufWriter = LegacyV05Writer;
