import { argv, stdout } from "node:process";
import { VBufInstance } from "./vbuf.ts";

const colors = {
	red: "\x1b[31m",
	green: "\x1b[32m",
	yellow: "\x1b[33m",
	blue: "\x1b[34m",
	purple: "\x1b[35m",
	gray: "\x1b[90m",
	cyan: "\x1b[36m",
	white: "\x1b[37m",
	reset: "\x1b[0m",
};

const path = argv[2];
if (!path) {
	console.log(`Use: bun debugger.ts <file.vbuf>`);
	process.exit(1);
}

const buffer = new Uint8Array(await Bun.file(path).arrayBuffer());
const view = new DataView(buffer.buffer);

const magic = new TextDecoder().decode(buffer.subarray(0, 4));
const version = view.getUint32(4, true);
const aShift = buffer[8]!;
const alignment = 1 << aShift;
const dataLen = view.getUint32(12, true);

console.log(`${colors.blue}${magic} v${version.toString(16)} | Alignment: ${alignment} B (AShift: ${aShift}) | Total Len: ${dataLen}${colors.reset}`);
console.log("-".repeat(100));

let i = 16;
while (i + 8 <= buffer.length) {
	const anchor = view.getBigUint64(i, true);

	if (anchor === 0n) {
		i += 8;
		continue;
	}

	const id = Number((anchor >> 16n) & 0xFFFFn);
	const plen = Number((anchor >> 32n) & 0xFFFFn);
	const hasOverflow = (anchor & (1n << 9n)) !== 0n;

	let count: bigint;
	let headerSize: number;

	if (hasOverflow) {
		count = view.getBigUint64(i + 8, true);
		headerSize = 16;
	} else {
		count = (anchor >> 48n) & 0xFFFFn;
		headerSize = 8;
	}

	const dataStart = (i + headerSize + (alignment - 1)) & ~(alignment - 1);
	const dataBytes = Number(count) * (plen / 8);

	console.log(`${colors.yellow}Anchor (0x${anchor.toString(16).toUpperCase()})${colors.reset} | ${colors.cyan}ID: ${id.toString().padEnd(6)}${colors.reset} | ${colors.green}Count: ${count.toString().padEnd(10)}${colors.reset} | ${colors.purple}${plen}-bit${colors.reset} | Offset: ${i}`);

	i = dataStart + dataBytes;
	i = (i + 7) & ~7;
}
