const MAGIC = 0x46554256; // "VBUF"
const VERSION = 0x00050000; // 0.5.0

export class VBufInstance {
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

export class VBufWriter {
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
