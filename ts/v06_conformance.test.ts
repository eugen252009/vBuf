import { describe, expect, test } from "bun:test";
import path from "node:path";
import { VBufV06, VBufV06Writer, V06ValidationError } from "./vbuf.ts";

const fixtureDir = path.join(import.meta.dir, "../tests/fixtures/v06");
const readFixture = async (name: string) =>
	new Uint8Array(await Bun.file(path.join(fixtureDir, name)).arrayBuffer());

type Manifest = { files: Array<{ path: string; accept: boolean }> };

const manifest = await Bun.file(path.join(fixtureDir, "manifest.json")).json() as Manifest;

describe("canonical vBuf v0.6 checked reader", () => {
	test("all cross-language fixtures have the declared outcome", async () => {
		expect(manifest.files.length).toBe(42);
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

	test("validated physical and payload ranges refine without escaping provenance", async () => {
		const parsed = new VBufV06(await readFixture("valid-basic.vbuf"));
		const physical = parsed.blockRange(0);
		expect([physical.offset, physical.length, physical.end]).toEqual([24n, 20n, 44n]);
		expect(Array.from(physical.bytes())).toEqual(Array.from(parsed.mem.slice(24, 44)));
		const payload = parsed.payloadRange(0);
		expect([payload.offset, payload.length, payload.end]).toEqual([32n, 12n, 44n]);
		expect(Array.from(payload.refine(4n, 8n).bytes())).toEqual(Array.from(parsed.mem.slice(36, 44)));
		expect(() => payload.refine(12n, 1n)).toThrow(V06ValidationError);
		expect(() => payload.refine((1n << 64n) - 1n, 1n)).toThrow(V06ValidationError);
		payload.requireAlignment(8n);
		expect(() => payload.requireAlignment(3n)).toThrow(V06ValidationError);
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
		expect(parsed.blocks.map((block) => block.continuation)).toEqual([true, true, false]);
		expect(Array.from(parsed.getU8(7, 0))).toEqual([65]);
		expect(Array.from(parsed.getU8(7, 1))).toEqual([66]);
		expect(Array.from(parsed.getU8(7, 2))).toEqual([67]);
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

	test("portable writer matches the Rust/C canonical fixture", async () => {
		const writer = new VBufV06Writer(4);
		writer.writeU8({ keyId: 10 }, new Uint8Array([1, 2, 255]));
		writer.writeU16({ keyId: 11 }, new Uint16Array([0x1234, 0xabcd]));
		writer.writeU32({ keyId: 12 }, new Uint32Array([1, 0xdeadbeef]));
		writer.writeU64({ keyId: 13 }, new BigUint64Array([1n, 0x0102030405060708n]));
		writer.writeI8({ keyId: 14 }, new Int8Array([-1, 2]));
		writer.writeI16({ keyId: 15 }, new Int16Array([-2, 3]));
		writer.writeI32({ keyId: 16 }, new Int32Array([-3, 4]));
		writer.writeI64({ keyId: 17 }, new BigInt64Array([-4n, 5n]));
		writer.writeF32({ keyId: 18 }, new Float32Array([1.5, -2.25]));
		writer.writeF64({ keyId: 19 }, new Float64Array([3.5, -4.75]));
		writer.writeOpaque({ keyId: 20 }, new Uint8Array([97, 98, 99]));
		writer.writeU32({ keyId: 21, physical: 0 }, new Uint32Array([42]));
		writer.writeOpaque({ keyId: 22, continuation: true }, new Uint8Array([65]));
		writer.writeOpaque({ keyId: 22 }, new Uint8Array([66]));
		const bytes = writer.finish();
		expect(Array.from(bytes)).toEqual(Array.from(await readFixture("valid-writer-primitives.vbuf")));
		const direct = new VBufV06(bytes);
		expect(Array.from(direct.getU16(11))).toEqual([0x1234, 0xabcd]);
		expect(Array.from(direct.getI64(17))).toEqual([-4n, 5n]);
		expect(Array.from(direct.getF32(18))).toEqual([1.5, -2.25]);
		expect(Array.from(direct.getOpaque(20))).toEqual([97, 98, 99]);
	});

	test("writer handles extended counts, indefinite streams, and continuation errors", () => {
		const extended = new VBufV06Writer(3);
		extended.writeU8({ keyId: 1 }, new Uint8Array(65536));
		const extendedBytes = extended.finish();
		expect(new VBufV06(extendedBytes).blocks[0]!.count).toBe(65536n);

		const indefinite = new VBufV06Writer(3, true);
		indefinite.writeU16({ keyId: 1 }, new Uint16Array([1, 2]));
		const indefiniteBytes = indefinite.finish();
		expect(indefiniteBytes[9]).toBe(1);
		expect(new DataView(indefiniteBytes.buffer).getBigUint64(16, true)).toBe(0n);
		expect(new VBufV06(indefiniteBytes).indefinite).toBeTrue();

		const chain = new VBufV06Writer(3);
		chain.writeU8({ keyId: 7, continuation: true }, new Uint8Array([1]));
		expect(() => chain.writeU8({ keyId: 8 }, new Uint8Array([2]))).toThrow(V06ValidationError);
		chain.writeU8({ keyId: 7 }, new Uint8Array([2]));
		expect(() => new VBufV06(chain.finish())).not.toThrow();

		const unterminated = new VBufV06Writer(3);
		unterminated.writeU8({ keyId: 7, continuation: true }, new Uint8Array([1]));
		expect(() => unterminated.finish()).toThrow(V06ValidationError);
	});
});
