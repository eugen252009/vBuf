// import xxhash from 'xxhash-wasm';
import fs from 'fs';
import readline from 'readline';

async function runVerification() {
	// const { h64 } = await xxhash();
	const VBUF_FILE = '../rust/output.vbuf';
	const JSON_FILE = '../rust/openfoodfacts-products.jsonl';

	const jsonLines = [];
	const rl = readline.createInterface({
		input: fs.createReadStream(JSON_FILE),
		terminal: false
	});

	for await (const line of rl) {
		if (line.trim()) {
			jsonLines.push(JSON.parse(line));
			if (jsonLines.length >= 200_000) break;
		}
	}
	rl.close();

	const buffer = Buffer.alloc(3 * 1024 * 1024 * 1024);
	const fd = fs.openSync(VBUF_FILE, 'r');
	fs.readSync(fd, buffer, 0, buffer.length, 0);
	fs.closeSync(fd);

	let offset = 0;
	let vbufProducts = [];
	let currentProduct = null;

	while (offset + 16 <= buffer.length) {
		const hashInHeader = buffer.readUInt32LE(offset);
		// const meta = buffer.readUInt8(offset + 4);
		const payloadLen = buffer.readUInt16LE(offset + 6);

		const payload = buffer.toString('utf8', offset + 8, offset + 8 + payloadLen).replace(/\0/g, '');

		if (payload.length >= 8 && /^\d+$/.test(payload) || hashInHeader === 0xd667df67) {
			if (currentProduct) vbufProducts.push(currentProduct);
			currentProduct = { id: payload, fields: new Map() };
		}

		if (currentProduct) {
			currentProduct.fields.set(hashInHeader, payload);
		}

		const currentSize = 8 + payloadLen;
		const padding = (16 - (currentSize % 16)) % 16;
		offset += currentSize + padding;
	}
	if (currentProduct) vbufProducts.push(currentProduct);

	jsonLines.forEach((json) => {
		const vbuf = vbufProducts.find(p => p.id === json._id);

		if (vbuf) {
			if (!vbuf.fields.has(0xd667df67)) {
				console.log(`   Struktur NIO (ID-Hash matcht)`);
			}
		} else {
			console.log(`❌ Produkt ${json._id} fehlt im VBuf-Extrakt.`);
		}
	});
}

runVerification().catch(console.error);
