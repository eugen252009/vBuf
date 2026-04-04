import { vBuf } from "./vbuf.ts";

(BigInt.prototype as any).toJSON = function() {
	return this.toString();
};
// const input = Bun.mmap("./data.json");
const input = Bun.mmap("./out");


const vb = new vBuf(input);
// const binaryOutput = vb.serialize();

// console.log(JSON.stringify(vb.toObject()))
// await Bun.write("./data.vbuf", binaryOutput);
