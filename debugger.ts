import { argv, stdout } from "node:process";

const SemNames: Record<number, string> = {
	0: "NUM", 1: "STR", 2: "BOL", 3: "NUL", 4: "ARR", 5: "OBJ", 6: "UUID"
};
const PhysNames: Record<number, string> = {
	0: "NUL ", 1: "SMI ", 2: "BIGI", 3: "BLOB", 4: "UUID", 5: "FLOT", 6: "FAL", 7: "TRU"
};

function formatPhysically(pType: number, data: Uint8Array): string {
	if (data.length === 0 && pType === 0) return "null";
	const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
	try {
		switch (pType) {
			case 0: return "null";
			case 1: return dv.getInt32(0, true).toString();
			case 2: return dv.getBigInt64(0, true).toString();
			case 3: return new TextDecoder().decode(data);
			case 4: return data.toHex().replace(/(.{8})(.{4})(.{4})(.{4})(.{12})/, "$1-$2-$3-$4-$5");
			case 5: return dv.getFloat64(0, true).toString();
			case 6: return "FALSE";
			case 7: return "TRUE";
			default: return "???";
		}
	} catch { return "ERR_DECODE"; }
}

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

console.log(`${colors.gray}HEADER (T|L|V)${colors.reset} | ${colors.red}TYPE-MAP${colors.reset} | ${colors.cyan}KEY/PATH${colors.reset} | ${colors.green}VALUE${colors.reset}\n`);

const magic = new TextDecoder().decode(buffer.subarray(0, 4));
const version = buffer[4];
const dataLen = view.getBigUint64(8, true);
console.log(`${colors.blue}${magic} v${version} | Total Len: ${dataLen}${colors.reset}`);
console.log("-".repeat(100));

let i = 16;
while (i < buffer.length) {
	const combinedType = buffer[i]!;
	const keyLen = buffer[i + 1]!;
	const valLen = view.getUint16(i + 2, true);

	const sType = (combinedType >> 4) & 0x0F;
	const pType = combinedType & 0x0F;

	const hT = combinedType.toString(16).padStart(2, '0');
	const hK = keyLen.toString(16).padStart(2, '0');
	const hV = valLen.toString(16).padStart(4, '0');
	const headerStr = `${colors.red}${hT}${colors.yellow}${hK}${hV}${colors.reset}`;

	const keyStart = i + 4;
	const keyName = new TextDecoder().decode(buffer.subarray(keyStart, keyStart + keyLen));
	const valBuf = buffer.subarray(keyStart + keyLen, keyStart + keyLen + valLen);

	const typeLabel = `${SemNames[sType] || '?'}->${PhysNames[pType] || '?'}`;
	const readableVal = formatPhysically(pType, valBuf);

	const minSize = 4 + keyLen + valLen + 4;
	const totalSize = Math.ceil(minSize / 16) * 16;
	const crc = buffer.subarray(i + totalSize - 4, i + totalSize).toHex();

	stdout.write(`${headerStr} | `);
	stdout.write(`${colors.red}${typeLabel.padEnd(12)}${colors.reset} | `);
	stdout.write(`${colors.cyan}${keyName.padEnd(12)}${colors.reset} | `);
	stdout.write(`${colors.green}${readableVal.padEnd(25)}${colors.reset} `);
	stdout.write(`${colors.gray}CRC:${crc}${colors.reset}\n`);

	i += totalSize;
}
