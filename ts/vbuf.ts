import { stdout } from "process"

const magic = 0x46554256	// "VBUF"
export const enum TokenType {
	LCurly = "LCurly",     // {
	RCurly = "RCurly",     // }
	LBracket = "LBracket",   // [
	RBracket = "RBracket",   // ]
	Colon = "Colon",      // :
	Comma = "Comma",      // ,
	String = "String",     // "..."
	Integer = "Integer",     // 123 / -1.23 / 1e10
	Float = "Float",     // 123 / -1.23 / 1e10
	SMI = "SMI",     // 123 / -1.23 / 1e10
	BigInt = "BigInt", // 1234567890123456789
	SMIString = "SMIString",     // 123 / -1.23 / 1e10
	BigIntString = "BigIntString",
	True = "True",       // true
	False = "False",      // false
	Null = "Null",       // null
	EOF = "EOF",
	UUID = "UUID",
}
export enum VBufType {
	Null = 0x00,
	SMI = 0x01,      // Small Integer (i32)
	BigInt = 0x02,   // 64-bit oder größer
	String = 0x03,   // UTF-8
	UUID = 0x04,     // 16 Bytes Raw
	Float = 0x05,    // 64-bit IEEE
	Array = 0x06,    // Metadaten für Listen (optional bei Flat-Ansatz)
	Bool = 0x07      // True/False
}
type Token = {
	type: TokenType
	start: number
	length: number
	value?: any
}
const CRC32_TABLE = new Uint32Array(256).map((_, i) => {
	let c = i;
	for (let k = 0; k < 8; k++) {
		c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
	}
	return c;
});
enum Sem { NUM = 0, STR = 1, BOL = 2, NUL = 3, ARR = 4, OBJ = 5, UUID = 6 }
enum Phys { NUL = 0, SMI = 1, BIGI = 2, BLOB = 3, UUID = 4, FLOT = 5, TRU = 6, FAL = 7 }

export class vBuf {
	private view: Uint8Array;
	private i = 0;
	private len: number;
	private cells: Uint8Array[] = [];
	public debug = false;

	constructor(input: ArrayBufferLike | Uint8Array) {
		this.view = input instanceof Uint8Array ? input : new Uint8Array(input);
		this.len = this.view.length;

		const obj = this.parseToJson();

		const flatMap = new Map<string, any>();
		this.flatten(obj, "", flatMap);

		this.debug && console.log(flatMap)

		for (const [path, value] of flatMap) {
			this.addManualCell(path, value);
		}
	}
	private processData(data: any, prefix: string = "") {
		for (const [key, value] of Object.entries(data)) {
			const fullPath = prefix ? `${prefix}.${key}` : key;

			if (Array.isArray(value)) {

				value.forEach((val, index) => {
					this.processData({ [index]: val }, fullPath);
				});
			} else if (value !== null && typeof value === 'object') {

				this.processData(value, fullPath);
			} else {

				this.addManualCell(fullPath, value);
			}
		}
	}
	private addManualCell(path: string, value: any) {
		let sType: Sem;
		let pType: Phys;
		let payload: Uint8Array = new Uint8Array(0);

		if (value === null || value === undefined) {
			return this.cells.push(this._createCell(3, 0, path, payload));
		}

		switch (typeof value) {
			case "boolean":
				sType = Sem.BOL;
				pType = value ? Phys.TRU : Phys.FAL;
				break;

			case "string":
				sType = Sem.STR;
				if (this.isUUID(value)) {
					pType = Phys.UUID;
					payload = this._packUUID(value);
				} else {
					const num = Number(value);
					if (value.length > 0 && value.length < 11 && !isNaN(num) && Number.isInteger(num) && String(num) === value && num >= -2147483648 && num <= 2147483647) {
						pType = Phys.SMI;
						payload = new Uint8Array(4);
						new DataView(payload.buffer).setInt32(0, num, true);
					} else {
						pType = Phys.BLOB;
						payload = new TextEncoder().encode(value);
					}
				}
				break;

			case "number":
				sType = Sem.NUM;
				if (Number.isInteger(value) && value >= -2147483648 && value <= 2147483647) {
					pType = Phys.SMI;
					payload = new Uint8Array(4);
					new DataView(payload.buffer).setInt32(0, value, true);
				} else {
					pType = Phys.FLOT;
					payload = new Uint8Array(8);
					new DataView(payload.buffer).setFloat64(0, value, true);
				}
				break;

			case "bigint":
				sType = Sem.NUM;
				if (value <= 9223372036854775807n && value >= -9223372036854775808n) {
					pType = Phys.BIGI;
					payload = new Uint8Array(8);
					new DataView(payload.buffer).setBigInt64(0, value, true);
				} else {
					pType = Phys.BLOB;
					payload = new TextEncoder().encode(value.toString());
				}
				break;

			default:
				console.log(typeof value, "not supported");
				return;
		}

		this.cells.push(this._createCell(sType, pType, path, payload));
	}
	private computeCRC(data: Uint8Array): number {
		let crc = 0 ^ -1;
		for (let i = 0; i < data.length; i++) {
			crc = (crc >>> 8) ^ CRC32_TABLE[(crc ^ data[i]!) & 0xFF]!;
		}
		return (crc ^ -1) >>> 0;
	}
	private _createCell(sType: number, pType: number, key: string, payload: Uint8Array): Uint8Array {
		const keyBuf = new TextEncoder().encode(key);
		const keyLen = keyBuf.length;
		const valLen = payload.length;

		const minSize = 4 + keyLen + valLen + 4;
		const totalSize = Math.ceil(minSize / 16) * 16;

		const cell = new Uint8Array(totalSize);
		const view = new DataView(cell.buffer);

		cell[0] = (sType << 4) | (pType & 0x0F);
		cell[1] = keyLen;
		view.setUint16(2, valLen, true);

		cell.set(keyBuf, 4);

		cell.set(payload, 4 + keyLen);

		const crcValue = this.computeCRC(cell.subarray(0, 4 + keyLen + valLen));
		view.setUint32(totalSize - 4, crcValue, true);

		return cell;
	}

	public serialize(): Uint8Array {
		const totalDataLen = this.cells.reduce((sum, c) => sum + c.length, 0);
		const output = new Uint8Array(16 + totalDataLen);
		const dv = new DataView(output.buffer);

		dv.setUint32(0, magic, true);
		dv.setUint8(4, 1);
		dv.setBigUint64(8, BigInt(totalDataLen), true);

		let offset = 16;
		for (const cell of this.cells) {
			output.set(cell, offset);
			offset += cell.length;
		}
		return output;
	}
	private parseToJson(): any {
		const token = this.next();
		switch (token.type) {
			case TokenType.LCurly: return this.parseObject();
			case TokenType.LBracket: return this.parseArray();
			case TokenType.Null: return null;
			case TokenType.True: return true;
			case TokenType.False: return false;
		}
		return token.value ?? this.getTokenString(token);
	}

	private parseObject(): any {
		const obj: any = {};
		while (true) {
			const token = this.next();
			if (token.type === TokenType.RCurly) break;
			if (token.type === TokenType.Comma) continue;

			const key = this.getTokenString(token);
			this.next();
			obj[key] = this.parseToJson();
		}
		return obj;
	}

	private parseArray(): any[] {
		const arr: any[] = [];
		while (true) {
			const token = this.next();
			if (token.type === TokenType.RBracket) break;
			if (token.type === TokenType.Comma) continue;

			this.i = token.start - (token.type === TokenType.String ? 1 : 0);
			arr.push(this.parseToJson());
		}
		return arr;
	}


	private isUUID(str: string): boolean {
		if (str.length !== 36) return false;
		return /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(str);
	}
	_packUUID(uuidStr: string): Uint8Array {
		const hex = uuidStr.replace(/-/g, "");
		if (hex.length !== 32) throw new Error("Invalid UUID length");

		const buf = new Uint8Array(16);
		for (let i = 0; i < 16; i++) {
			buf[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
		}
		return buf;
	}
	flatten(obj: any, prefix = "", result: Map<string, any> = new Map()) {
		if (obj === null || typeof obj !== "object") {
			result.set(prefix, obj);
			return result;
		}

		for (const key in obj) {
			const path = prefix
				? (Array.isArray(obj) ? `${prefix}[${key}]` : `${prefix}.${key}`)
				: key;

			this.flatten(obj[key], path, result);
		}

		return result;
	}

	private getTokenString(token: Token): string {
		return new TextDecoder().decode(this.view.subarray(token.start, token.start + token.length));
	}
	debugToken(token: Token) {
		stdout.write(`${token.type} ${token.start} ${token.length}`)
		if (token.value)
			stdout.write(`${token.value}\n`)
		else
			stdout.write(`\n`)
	}
	private isWhitespace(c: number): boolean {
		return c === 0x20 || c === 0x0a || c === 0x0d || c === 0x09;
	}

	private skipWhitespace() {
		while (this.i < this.len && this.isWhitespace(this.view[this.i]!)) {
			this.i++;
		}
	}

	next(): Token {
		this.skipWhitespace();
		if (this.i >= this.len) {
			return { type: TokenType.EOF, start: this.i, length: 0 };
		}

		const start = this.i;
		const c = this.view[this.i]!;

		switch (c) {
			case 0x7b: this.i++; return { type: TokenType.LCurly, start, length: 1 };   // {
			case 0x7d: this.i++; return { type: TokenType.RCurly, start, length: 1 };   // }
			case 0x5b: this.i++; return { type: TokenType.LBracket, start, length: 1 }; // [
			case 0x5d: this.i++; return { type: TokenType.RBracket, start, length: 1 }; // ]
			case 0x3a: this.i++; return { type: TokenType.Colon, start, length: 1 };    // :
			case 0x2c: this.i++; return { type: TokenType.Comma, start, length: 1 };    // ,

			case 0x22: // "
			case 0x27: // '
				return this.readString(c);

			case 0x74: // 't' for true
				return this.consumeKeyword(start, "true", TokenType.True);
			case 0x66: // 'f' for false
				return this.consumeKeyword(start, "false", TokenType.False);
			case 0x6e: // 'n' for null
				return this.consumeKeyword(start, "null", TokenType.Null);

			default:
				// Numbers (0-9 oder -)
				if ((c >= 0x30 && c <= 0x39) || c === 0x2d) {
					return this.readNumber();
				}
				throw new Error(`Unexpected character: ${String.fromCharCode(c)} at index ${this.i} ${c}`);
		}
	}
	private consumeKeyword(start: number, expected: string, type: TokenType): Token {
		let value: any;
		switch (type) {
			case TokenType.True:
				value = true;
				break;
			case TokenType.False:
				value = false;
				break;
			case TokenType.Null:
				value = null;
				break;
		}

		for (let j = 0; j < expected.length; j++) {
			if (this.view[this.i + j] !== expected.charCodeAt(j)) {
				throw new Error(`Invalid keyword at ${start}. Expected ${expected}`);
			}
		}

		this.i += expected.length;
		return { type, start, length: expected.length, value };
	}

	private readString(delim: number): Token {
		const start = this.i + 1;
		let isNumeric = true;
		let val = 0n;
		this.i++;
		while (this.i < this.len) {
			const c = this.view[this.i]!;
			if (c === delim) {
				this.i++;
				break;
			}

			if (c >= 0x30 && c <= 0x39) {
				val = val * 10n + BigInt(c - 0x30);
			} else {
				isNumeric = false;
			}
			this.i++;
		}

		const length = (this.i - 1) - start;

		if (length === 36 && this.view[start + 8] === 0x2d && // '-'
			this.view[start + 13] === 0x2d && // '-'
			this.view[start + 18] === 0x2d && // '-'
			this.view[start + 23] === 0x2d && // '-'
			this.view[start + 14] === 0x34    // '4' (Version 4 UUID)
		) {
			return { type: TokenType.UUID, start, length: 36 };
		}
		if (isNumeric && length > 0) {
			if (val <= 2147483647n && val >= -2147483648n) {
				return { type: TokenType.SMIString, start, length, value: Number(val) };
			}
			return { type: TokenType.BigIntString, start, length, value: val };
		}

		return { type: TokenType.String, start, length };
	}

	private readNumber(): Token {
		const start = this.i;
		let val = 0;
		let bigVal: bigint | null = null;
		let isBigInt = false;
		let isNegative = false;
		let hasDecimal = false;

		if (this.view[this.i] === 0x2d) { // '-'
			isNegative = true;
			this.i++;
		} else if (this.view[this.i] === 0x2b) { // '+'
			this.i++;
		}

		while (this.i < this.len) {
			const c = this.view[this.i]!;

			if (c >= 0x30 && c <= 0x39) { // 0-9
				const digit = c - 0x30;

				if (!isBigInt) {
					if (val > (Number.MAX_SAFE_INTEGER - digit) / 10) {
						isBigInt = true;
						bigVal = BigInt(val) * 10n + BigInt(digit);
					} else {
						val = val * 10 + digit;
					}
				} else {
					bigVal = bigVal! * 10n + BigInt(digit);
				}

				this.i++;
				continue;
			} else if (c === 0x2e || c === 0x65 || c === 0x45) { // '.' oder 'e'/'E'
				hasDecimal = true;
				this.skipRestOfNumber();
				break;
			}
			break;
		}

		const length = this.i - start;

		if (hasDecimal) {
			return { type: TokenType.Float, start, length };
		}

		if (isNegative) val = -val;

		if (val >= -2147483648 && val <= 2147483647) {
			return { type: TokenType.SMI, start, length, value: val };
		}
		if (isBigInt)
			return { type: TokenType.BigInt, start, length, value: bigVal };
		return { type: TokenType.Integer, start, length, value: val };
	}

	private skipRestOfNumber() {
		while (this.i < this.len) {
			const c = this.view[this.i]!;
			if ((c >= 0x30 && c <= 0x39) || c === 0x2e || c === 0x2d || c === 0x2b || c === 0x65 || c === 0x45) {
				this.i++;
				continue;
			}
			break;
		}
	}
	private getKeyFromCell(cell: Uint8Array): string {
		const keyLen = cell[1]!;
		const keyBuf = cell.subarray(4, 4 + keyLen);
		return new TextDecoder().decode(keyBuf);
	}
	private getValueFromCell(cell: Uint8Array): any {
		const combinedType = cell[0]!;
		const pType = combinedType & 0x0F; // Physikalischer Typ (untere 4 Bits)
		const keyLen = cell[1]!;
		const valLen = new DataView(cell.buffer, cell.byteOffset, cell.byteLength).getUint16(2, true);

		// Start der Payload: Header (4) + Key (keyLen)
		const valStart = 4 + keyLen;
		const valBuf = cell.subarray(valStart, valStart + valLen);
		const dv = new DataView(valBuf.buffer, valBuf.byteOffset, valBuf.byteLength);

		switch (pType) {
			case Phys.NUL:
				return null;
			case Phys.SMI: // Phys.SMI (i32)
				return dv.getInt32(0, true);
			case Phys.BIGI: // Phys.BIGI (i64)
				return dv.getBigInt64(0, true);
			case Phys.BLOB: // Phys.BLOB (String oder riesiger BigInt)
				const str = new TextDecoder().decode(valBuf);
				// Check, ob es ein BigInt-Fallback war (endet auf n bei der Serialisierung)
				if ((combinedType >> 4) === 0 && valLen > 0 && !isNaN(Number(str)) && str.length > 15) {
					try { return BigInt(str); } catch { return str; }
				}
				return str;
			case Phys.UUID: // Phys.UUID
				return valBuf.toHex().replace(/(.{8})(.{4})(.{4})(.{4})(.{12})/, "$1-$2-$3-$4-$5");
			case Phys.FLOT: // Phys.FLOT (f64)
				return dv.getFloat64(0, true);
			case Phys.FAL: // Phys.FAL (Inlined False)
				return false;
			case Phys.TRU: // Phys.TRU (Inlined True)
				return true;
			default:
				return null;
		}
	}
	public toObject(): any {
		const root: any = {};

		for (const cell of this.cells) {
			const path = this.getKeyFromCell(cell);
			const value = this.getValueFromCell(cell);

			const parts = path.split(/\.|\[|\]/).filter(Boolean);
			let current = root;

			for (let i = 0; i < parts.length; i++) {
				const part = parts[i]!;
				const isLast = i === parts.length - 1;
				const nextPart = parts[i + 1]!;
				const nextIsArray = nextPart !== undefined && !isNaN(Number(nextPart));

				if (isLast) {
					current[part] = value;
				} else {
					if (!(part in current)) {
						current[part] = nextIsArray ? [] : {};
					}
					current = current[part];
				}
			}
		}
		return root;
	}
}
