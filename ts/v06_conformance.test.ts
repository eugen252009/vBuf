import { describe, expect, test } from "bun:test";
import path from "node:path";
import { VBufV06, V06ValidationError } from "./vbuf.ts";

const fixtureDir = path.join(import.meta.dir, "../tests/fixtures/v06");
const readFixture = async (name: string) =>
	new Uint8Array(await Bun.file(path.join(fixtureDir, name)).arrayBuffer());

type Manifest = { files: Array<{ path: string; accept: boolean }> };

const manifest = await Bun.file(path.join(fixtureDir, "manifest.json")).json() as Manifest;

describe("canonical vBuf v0.6 checked reader", () => {
	test("all cross-language fixtures have the declared outcome", async () => {
		expect(manifest.files.length).toBe(41);
		for (const fixture of manifest.files) {
			const bytes = await readFixture(fixture.path);
			let accepted = true;
			try {
				new VBufV06(bytes);
			} catch (error) {
				expect(error).toBeInstanceOf(V06ValidationError);
				accepted = false;
			}
			expect(accepted, fixture.path).toBe(fixture.accept);
		}
	});

	test("descriptor and exact range validation precede typed-array construction", async () => {
		const parsed = new VBufV06(await readFixture("valid-basic.vbuf"));
		expect(parsed.baseStep).toBe(8);
		expect(parsed.dataRegionStart).toBe(24);
		expect(parsed.blocks[0]).toMatchObject({
			blockStart: 24,
			payloadStart: 32,
			payloadLength: 12,
			payloadEnd: 44,
			keyId: 1,
			semantic: 0,
			physical: 1,
			bitWidth: 32,
			count: 3n,
		});
		expect(Array.from(parsed.getU32(1))).toEqual([1, 2, 3]);
		expect(() => parsed.getF32(1)).toThrow(V06ValidationError);
	});

	test("duplicate KeyIDs use explicit physical occurrence", async () => {
		const parsed = new VBufV06(await readFixture("valid-duplicate-chain.vbuf"));
		expect(parsed.blocks.map((block) => block.continuation)).toEqual([true, false]);
		expect(Array.from(parsed.getU8(7, 0))).toEqual([65]);
		expect(Array.from(parsed.getU8(7, 1))).toEqual([66]);
	});

	test("legal empty, zero-length, and partial-final regions are accepted", async () => {
		expect(new VBufV06(await readFixture("valid-empty.vbuf")).blocks).toHaveLength(0);
		expect(new VBufV06(await readFixture("valid-zero-array.vbuf")).getU8(2)).toHaveLength(0);
		const partial = new VBufV06(await readFixture("valid-basic.vbuf"));
		expect((partial.dataRegionStart + Number(partial.dataRegionSize)) % partial.baseStep).not.toBe(0);
	});

	test("typed arrays reject an unaligned underlying Uint8Array", async () => {
		const original = await readFixture("valid-basic.vbuf");
		const storage = new Uint8Array(original.byteLength + 1);
		storage.set(original, 1);
		const parsed = new VBufV06(new Uint8Array(storage.buffer, 1, original.byteLength));
		expect(() => parsed.getU32(1)).toThrow(V06ValidationError);
	});

	test("two declared Float64 values with one available never clamp", async () => {
		const truncated = await readFixture("payload-truncated-element.vbuf");
		expect(() => new VBufV06(truncated)).toThrow(V06ValidationError);
	});
});
