import { VBuf } from "./vbuf.ts";

const file = Bun.mmap("./data.json")
const fb = new VBuf(file)
