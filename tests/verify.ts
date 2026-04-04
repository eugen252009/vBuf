import xxhash from 'xxhash-wasm';
import fs from 'fs';
import readline from 'readline';

async function runVerification() {
	const { h64 } = await xxhash();
	const VBUF_FILE = './test_output.vbuf';
	const JSON_FILE = './test_input.json';

	// 1. Die ersten 20 Zeilen JSON laden
	const jsonLines = [];
	const rl = readline.createInterface({
		input: fs.createReadStream(JSON_FILE),
		terminal: false
	});

	for await (const line of rl) {
		if (line.trim()) {
			jsonLines.push(JSON.parse(line));
			if (jsonLines.length >= 20) break;
		}
	}
	rl.close();

	// 2. Den Anfang des VBuf laden (5MB)
	const buffer = Buffer.alloc(5 * 1024 * 1024);
	const fd = fs.openSync(VBUF_FILE, 'r');
	fs.readSync(fd, buffer, 0, buffer.length, 0);
	fs.closeSync(fd);

	// Hilfsfunktion: Berechnet den 32-bit Hash-Teil, den du im Header nutzt
	// const getHeaderHash = (path: string) => Number(h64(path) & 0xFFFFFFFFn);

	// const ID_HASH = getHeaderHash("_id");
	let offset = 0;
	let vbufProducts = [];
	let currentProduct = null;

	// 3. VBuf linear parsen und Produkte gruppieren
	while (offset + 16 <= buffer.length) {
		const hashInHeader = buffer.readUInt32LE(offset); // Die ersten 4 Bytes
		// const meta = buffer.readUInt8(offset + 4);
		const payloadLen = buffer.readUInt16LE(offset + 6);

		// Payload extrahieren
		const payload = buffer.toString('utf8', offset + 8, offset + 8 + payloadLen).replace(/\0/g, '');

		// DEBUG: Wenn die Payload wie eine EAN/ID aussieht (nur Zahlen, lang genug)
		// ODER wenn der Hash-Teil d667df67 ist (deine ID aus Zelle 0)
		if (payload.length >= 8 && /^\d+$/.test(payload) || hashInHeader === 0xd667df67) {
			if (currentProduct) vbufProducts.push(currentProduct);
			currentProduct = { id: payload, fields: new Map() };
			// console.log(`Gefunden im VBuf: Produkt ${payload} mit Hash ${hashInHeader.toString(16)}`);
		}

		if (currentProduct) {
			currentProduct.fields.set(hashInHeader, payload);
		}

		const currentSize = 8 + payloadLen;
		const padding = (16 - (currentSize % 16)) % 16;
		offset += currentSize + padding;
	}
	if (currentProduct) vbufProducts.push(currentProduct);

	// 4. Abgleich mit LOGS
	console.log(`\n=== Verifikation ===\n`);
	jsonLines.forEach((json) => {
		// Wir suchen jetzt einfach per String-Vergleich in unseren gesammelten VBuf-Blöcken
		const vbuf = vbufProducts.find(p => p.id === json._id);

		if (vbuf) {
			console.log(`✅ Match: ${json._id}`);
			// Teste ein Feld, das sicher da ist
			// Wir nutzen hier den Hash d667df67 direkt für die ID selbst zum Test
			if (vbuf.fields.has(0xd667df67)) {
				console.log(`   Struktur OK (ID-Hash matcht)`);
			}
		} else {
			console.log(`❌ Produkt ${json._id} fehlt im VBuf-Extrakt.`);
		}
	});
}

runVerification().catch(console.error);
