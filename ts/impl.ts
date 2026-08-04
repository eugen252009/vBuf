import { VBufInstance, VBufWriter } from "./vbuf.ts";

// 1. Create a column-based binary using VBufWriter
const writer = new VBufWriter(4096);
const ids = new Int32Array([10001, 10002, 10003]);
const prices = new Float64Array([99.99, 149.50, 19.99]);

writer.writeColumn(1, ids, 32);
writer.writeColumn(2, prices, 64);

const binary = writer.finish();
await Bun.write("./test.vbuf", binary);
console.log("Successfully wrote test.vbuf!");

// 2. Read it back using VBufInstance
const input = new Uint8Array(await Bun.file("./test.vbuf").arrayBuffer());
const vb = new VBufInstance(input);

const readIds = vb.getCol(1) as Int32Array;
const readPrices = vb.getCol(2) as Float64Array;

console.log("IDs:", Array.from(readIds));
console.log("Prices:", Array.from(readPrices));
