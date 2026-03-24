import { stdout } from "process"

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
type Token = {
	type: TokenType
	start: number
	length: number
	value?: any
}
export class VBuf {
	private view: Uint8Array;
	private i = 0;
	private len: number;
	private _buffer: ArrayBuffer | undefined;
	public value: string | number | null = null;
	public debug = true;

	constructor(input: ArrayBufferLike | Uint8Array) {
		this.view = input instanceof Uint8Array ? input : new Uint8Array(input);
		this.len = this.view.length;
		this._parse(undefined, this.len)
	}
	_parse(buf: ArrayBuffer | undefined, length: number) {
		if (buf) this._buffer = buf
		else this._buffer = new ArrayBuffer(length)
		let totallength = 0
		while (true) {
			let token = this.next()
			if (token.type === TokenType.EOF) break
			if (this.debug) this.debugToken(token)
			totallength += token.length
			switch (token.type) {
				// case TokenType.LCurly:
				// case TokenType.RCurly:
				// case TokenType.Integer:
				// case TokenType.BigInt:

				default:
					console.log(token.type, " parsing not Implemented yet")
			}
		}
	}
	debugToken(token: Token) {
		stdout.write(`${token.type} ${token.start} ${token.length}`)
		if (token.value)
			stdout.write(`${token.value}\n`)
		else
			stdout.write(`\n`)
	}
	export(): ArrayBufferLike {
		return this._buffer!
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
		if (this.i >= this.len) return { type: TokenType.EOF, start: this.i, length: 0 };

		const c = this.view[this.i]!;

		switch (c) {
			case 0x7b: this.i++; return { type: TokenType.LCurly, start: this.i, length: 1 } // {
			case 0x7d: this.i++; return { type: TokenType.RCurly, start: this.i, length: 1 } // }
			case 0x5b: this.i++; return { type: TokenType.LBracket, start: this.i, length: 1 } // [
			case 0x5d: this.i++; return { type: TokenType.RBracket, start: this.i, length: 1 } // ]
			case 0x3a: this.i++; return { type: TokenType.Colon, start: this.i, length: 1 } // :
			case 0x2c: this.i++; return { type: TokenType.Comma, start: this.i, length: 1 } // ,

			case 0x27: // '
				return this.readString(0x27);
			case 0x22: // "
				return this.readString(0x22);

			case 0x74: // true
				return this.readLiteral("true", TokenType.True);

			case 0x66: // false
				return this.readLiteral("false", TokenType.False);

			case 0x6e: // null
				return this.readLiteral("null", TokenType.Null);

			default:
				if (c === 0x2d || (c >= 0x30 && c <= 0x39)) {
					return this.readNumber();
				}
				throw new Error(`Unexpected char: ${String.fromCharCode(c)}`);
		}
	}

	private readLiteral(str: string, toktype: TokenType): Token {
		const start = this.i
		for (let j = 0; j < str.length; j++) {
			if (this.view[this.i + j]! !== str.charCodeAt(j)) {
				throw new Error(`Invalid literal: expected ${str}`);
			}
		}
		this.i += str.length;
		return { type: toktype, start, length: str.length, value: toktype };
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

		if (length === 36 &&
			this.view[start + 8] === 0x2d && // '-'
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
}



