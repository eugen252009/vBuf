import { vBuf } from "./vbuf.ts";

const input = Bun.mmap("./data.json");

const vb = new vBuf(input);
const binaryOutput = vb.serialize();

await Bun.write("./data.vbuf", binaryOutput);
console.log("vBuf Datei erfolgreich erstellt!");
