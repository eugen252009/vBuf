import { describe, expect, test } from "bun:test";
import path from "node:path";
import { VBufInstance, VBufWriter } from "./vbuf.ts";

const fixturePath = (name: string) =>
	path.join(import.meta.dir, "../tests/fixtures/v05", name);

const readFixture = async (name: string) =>
	new Uint8Array(await Bun.file(fixturePath(name)).arrayBuffer());

describe("legacy v0.5 byte and behavior evidence", () => {
	test("current TypeScript writer bytes are preserved", async () => {
		const writer = new VBufWriter(16);
		writer.writeColumn(1, new Int32Array([1, 2, 3]), 32);
		writer.writeColumn(2, new Float64Array([1.5, -2.25]), 64);

		const expected = await readFixture("current-typescript.vbuf");
		expect(Array.from(writer.finish())).toEqual(Array.from(expected));
	});

	test("TypeScript reader reads Rust and TypeScript fixtures", async () => {
		const rust = new VBufInstance(await readFixture("current-rust.vbuf"));
		expect(Array.from(rust.getCol(1) as Int32Array)).toEqual([1, 2, 3]);
		expect(Array.from(rust.getCol(2) as Uint16Array)).toEqual([500, 1000]);

		const typescript = new VBufInstance(await readFixture("current-typescript.vbuf"));
		expect(Array.from(typescript.getCol(1) as Int32Array)).toEqual([1, 2, 3]);
		expect(Array.from(typescript.getCol(2) as Float64Array)).toEqual([1.5, -2.25]);
	});

	test("TypeScript legacy malformed behavior is preserved as evidence", async () => {
		const short = await readFixture("short.vbuf");
		const badMagic = await readFixture("bad-magic.vbuf");
		expect(() => new VBufInstance(short)).toThrow();
		expect(() => new VBufInstance(badMagic)).toThrow();

		// The header declares two Float64 values but only one remains. Legacy
		// subarray clamping returns a partial typed view. Step 4 must replace
		// this observed behavior with mandatory v0.6 rejection before view creation.
		const truncated = new VBufInstance(await readFixture("truncated-typescript.vbuf"));
		const observed = truncated.getCol(2) as Float64Array;
		expect(observed).toBeInstanceOf(Float64Array);
		expect(Array.from(observed)).toEqual([1.5]);
	});
});
