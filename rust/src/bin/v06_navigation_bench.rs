//! Experimental Step 5A generic BaseStep/Nano qualification benchmark.
//! Nano and directories here are in-memory benchmark artifacts, not wire format.

use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Cursor;
use std::time::Instant;
use vbuf_core::v06::{V06Physical, parse_v06};
use vbuf_core::writer::{BlockOptions, VBufV06Writer};

const SAMPLES: usize = 20;
const WARMUPS: usize = 3;
const CHECKPOINTS: [u64; 3] = [512, 4096, 65536];

#[derive(Clone, Copy)]
struct Plan {
    key: u16,
    width: u16,
    count: usize,
    semantic: u8,
    physical: V06Physical,
    continuation: bool,
    seed: u8,
}

struct Corpus {
    name: &'static str,
    plans: Vec<Plan>,
}

struct Sample {
    layout: &'static str,
    base_step: u64,
    operation: &'static str,
    variant: &'static str,
    sample: usize,
    nanos: u128,
    file_bytes: usize,
    payload_bytes: u64,
    block_count: usize,
    nano_bytes: usize,
    checkpoint_bytes: usize,
    directory_bytes: usize,
    checksum: u64,
}

fn corpora() -> Vec<Corpus> {
    let mut many_tiny = Vec::new();
    for i in 0..4096u16 {
        many_tiny.push(Plan {
            key: i % 64,
            width: 32,
            count: 2,
            semantic: 0,
            physical: V06Physical::Array,
            continuation: false,
            seed: i as u8,
        });
    }
    let mut few_large = Vec::new();
    for i in 0..8u16 {
        few_large.push(Plan {
            key: i,
            width: 32,
            count: 262144,
            semantic: 0,
            physical: V06Physical::Array,
            continuation: false,
            seed: i as u8,
        });
    }
    let mut mixed = many_tiny[..512].to_vec();
    mixed.extend(few_large.iter().copied());
    let mut aos = Vec::new();
    for record in 0..1024u16 {
        for field in 0..4u16 {
            aos.push(Plan {
                key: 100 + field,
                width: 32,
                count: 1,
                semantic: 0,
                physical: V06Physical::Scalar,
                continuation: false,
                seed: record as u8 ^ field as u8,
            });
        }
    }
    let soa = (0..4u16)
        .map(|field| Plan {
            key: 120 + field,
            width: 32,
            count: 1024,
            semantic: 0,
            physical: V06Physical::Array,
            continuation: false,
            seed: field as u8,
        })
        .collect();
    let mut composite = Vec::new();
    for i in 0..1024u16 {
        composite.push(Plan {
            key: 200 + i % 8,
            width: 8,
            count: 32,
            semantic: 3,
            physical: V06Physical::Array,
            continuation: true,
            seed: i as u8,
        });
        composite.push(Plan {
            key: 200 + i % 8,
            width: 8,
            count: 16,
            semantic: 3,
            physical: V06Physical::Array,
            continuation: true,
            seed: (i as u8).wrapping_add(1),
        });
        composite.push(Plan {
            key: 200 + i % 8,
            width: 8,
            count: 8,
            semantic: 3,
            physical: V06Physical::Array,
            continuation: false,
            seed: (i as u8).wrapping_add(2),
        });
    }
    let mut opaque = Vec::new();
    for i in 0..16u16 {
        opaque.push(Plan {
            key: 240 + i,
            width: 8,
            count: 524288,
            semantic: 3,
            physical: V06Physical::Array,
            continuation: false,
            seed: i as u8,
        });
    }
    vec![
        Corpus {
            name: "many-tiny",
            plans: many_tiny,
        },
        Corpus {
            name: "few-large",
            plans: few_large,
        },
        Corpus {
            name: "mixed",
            plans: mixed,
        },
        Corpus {
            name: "aos",
            plans: aos,
        },
        Corpus {
            name: "soa",
            plans: soa,
        },
        Corpus {
            name: "composite-continuation",
            plans: composite,
        },
        Corpus {
            name: "large-opaque",
            plans: opaque,
        },
        Corpus {
            name: "zero-and-partial",
            plans: vec![
                Plan {
                    key: 250,
                    width: 8,
                    count: 0,
                    semantic: 0,
                    physical: V06Physical::Array,
                    continuation: false,
                    seed: 0,
                },
                Plan {
                    key: 251,
                    width: 32,
                    count: 3,
                    semantic: 0,
                    physical: V06Physical::Array,
                    continuation: false,
                    seed: 1,
                },
            ],
        },
    ]
}

fn payload(plan: Plan) -> Vec<u8> {
    let bytes = plan.count.checked_mul(plan.width as usize / 8).unwrap();
    (0..bytes)
        .map(|i| plan.seed.wrapping_add(i as u8))
        .collect()
}

fn make_file(corpus: &Corpus, base_shift: u8) -> Vec<u8> {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), base_shift).unwrap();
    for plan in &corpus.plans {
        let options = BlockOptions {
            key_id: plan.key,
            physical: plan.physical,
            continuation: plan.continuation,
            payload_shift: 0,
        };
        writer
            .write_block(
                options,
                match plan.semantic {
                    0 => vbuf_core::v06::V06Semantic::Unsigned,
                    3 => vbuf_core::v06::V06Semantic::Opaque,
                    _ => unreachable!(),
                },
                plan.width,
                plan.count as u64,
                &payload(*plan),
            )
            .unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn payload_sum(bytes: &[u8], block: &vbuf_core::v06::V06Block) -> u64 {
    bytes[block.payload_start as usize..block.payload_end as usize]
        .iter()
        .map(|b| *b as u64)
        .sum()
}

fn checked_nano_len(data_size: u64, base_step: u64) -> Result<usize, &'static str> {
    if base_step == 0 || !base_step.is_power_of_two() {
        return Err("invalid BaseStep");
    }
    let slots = data_size
        .checked_add(base_step - 1)
        .ok_or("slot-count overflow")?
        / base_step;
    usize::try_from(slots.div_ceil(8)).map_err(|_| "Nano length exceeds host usize")
}

fn nano(
    blocks: &[vbuf_core::v06::V06Block],
    data_start: u64,
    data_size: u64,
    base_step: u64,
) -> Vec<u8> {
    let mut bits = vec![0u8; checked_nano_len(data_size, base_step).unwrap()];
    for block in blocks {
        let slot = (block.block_start - data_start) / base_step;
        bits[usize::try_from(slot / 8).unwrap()] |= 1 << (slot % 8);
    }
    bits
}

fn bit_is_set(bits: &[u8], slot: u64) -> bool {
    let byte = usize::try_from(slot / 8).unwrap();
    byte < bits.len() && bits[byte] & (1 << (slot % 8)) != 0
}

fn set_slots(bits: &[u8]) -> Vec<u64> {
    bits.iter()
        .enumerate()
        .flat_map(|(byte, value)| {
            (0..8).filter_map(move |bit| {
                ((*value & (1 << bit)) != 0).then_some((byte * 8 + bit) as u64)
            })
        })
        .collect()
}

fn checkpoints(bits: &[u8], slot_count: u64, interval: u64) -> Vec<(u64, u64)> {
    let mut result = Vec::new();
    let mut rank = 0u64;
    let mut slot = 0u64;
    while slot < slot_count {
        result.push((slot, rank));
        let end = (slot + interval).min(slot_count);
        while slot < end {
            if bit_is_set(bits, slot) {
                rank += 1;
            }
            slot += 1;
        }
    }
    result
}

fn raw_select(bits: &[u8], slot_count: u64, ordinal: u64) -> Option<u64> {
    let mut rank = 0u64;
    for slot in 0..slot_count {
        if bit_is_set(bits, slot) {
            if rank == ordinal {
                return Some(slot);
            }
            rank += 1;
        }
    }
    None
}

fn select_checkpointed(
    bits: &[u8],
    checkpoints: &[(u64, u64)],
    slot_count: u64,
    interval: u64,
    ordinal: u64,
) -> Option<(u64, u64)> {
    let &(start_slot, start_rank) = checkpoints
        .iter()
        .take_while(|(_, rank)| *rank <= ordinal)
        .last()?;
    let end_slot = (start_slot + interval).min(slot_count);
    let mut rank = start_rank;
    for slot in start_slot..end_slot {
        if bit_is_set(bits, slot) {
            if rank == ordinal {
                let next = ((slot + 1)..slot_count)
                    .find(|candidate| bit_is_set(bits, *candidate))
                    .unwrap_or(slot_count);
                return Some((slot, next));
            }
            rank += 1;
        }
    }
    None
}

fn serialize_nano(bits: &[u8]) -> Vec<u8> {
    bits.to_vec()
}

fn deserialize_nano(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

fn nano_checksum(bits: &[u8]) -> u64 {
    bits.iter()
        .enumerate()
        .map(|(i, byte)| (*byte as u64) << (i % 8))
        .sum()
}

fn nano_start_sum(bits: &[u8], slot_count: u64, data_start: u64, base_step: u64) -> u64 {
    (0..slot_count)
        .filter(|slot| bit_is_set(bits, *slot))
        .map(|slot| data_start + slot * base_step)
        .sum()
}

fn lookup_range(blocks: &[vbuf_core::v06::V06Block], key: u16) -> Option<(u64, u64, u64)> {
    blocks
        .iter()
        .find(|block| block.key_id == key)
        .map(|block| (block.block_start, block.payload_start, block.payload_len))
}

fn range_checksum(range: Option<(u64, u64, u64)>) -> u64 {
    range
        .map(|(a, b, c)| a ^ b.rotate_left(17) ^ c.rotate_left(31))
        .unwrap_or(u64::MAX)
}

fn make_directory(blocks: &[vbuf_core::v06::V06Block]) -> Vec<(u16, u64, u64, u64)> {
    let mut directory: Vec<_> = blocks
        .iter()
        .map(|b| (b.key_id, b.block_start, b.payload_start, b.payload_len))
        .collect();
    directory.sort_unstable();
    directory
}

fn directory_checksum(directory: &[(u16, u64, u64, u64)], key: u16) -> u64 {
    range_checksum(
        directory
            .iter()
            .find(|entry| entry.0 == key)
            .map(|entry| (entry.1, entry.2, entry.3)),
    )
}

fn measure<F: FnMut() -> u64>(mut f: F) -> Vec<u128> {
    for _ in 0..WARMUPS {
        black_box(f());
    }
    (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            black_box(f());
            start.elapsed().as_nanos()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn append_samples(
    out: &mut Vec<Sample>,
    layout: &'static str,
    base_step: u64,
    operation: &'static str,
    variant: &'static str,
    times: Vec<u128>,
    file: usize,
    payload: u64,
    blocks: usize,
    nano_bytes: usize,
    checkpoint_bytes: usize,
    directory_bytes: usize,
    checksum: u64,
) {
    out.extend(times.into_iter().enumerate().map(|(sample, nanos)| Sample {
        layout,
        base_step,
        operation,
        variant,
        sample,
        nanos,
        file_bytes: file,
        payload_bytes: payload,
        block_count: blocks,
        nano_bytes,
        checkpoint_bytes,
        directory_bytes,
        checksum,
    }));
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nano_matches_every_canonical_physical_start_for_all_layouts_and_bases() {
        for base_shift in [3u8, 4, 5, 6, 7, 8] {
            let base_step = 1u64 << base_shift;
            for corpus in corpora() {
                let bytes = make_file(&corpus, base_shift);
                let parsed = parse_v06(&bytes).unwrap();
                let bits = nano(
                    parsed.blocks(),
                    parsed.header().data_region_start,
                    parsed.header().data_region_size,
                    base_step,
                );
                let slots = set_slots(&bits);
                let expected: Vec<u64> = parsed
                    .blocks()
                    .iter()
                    .map(|block| {
                        (block.block_start - parsed.header().data_region_start) / base_step
                    })
                    .collect();
                assert_eq!(slots, expected, "{} BaseStep={base_step}", corpus.name);
                let slot_count = parsed.header().data_region_size.div_ceil(base_step);
                if let Some(last) = bits.last() {
                    let used = (slot_count % 8) as u8;
                    if used != 0 {
                        assert_eq!(*last & !((1u8 << used) - 1), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn continuation_members_remain_independent_nano_starts() {
        let corpus = corpora()
            .into_iter()
            .find(|corpus| corpus.name == "composite-continuation")
            .unwrap();
        let bytes = make_file(&corpus, 4);
        let parsed = parse_v06(&bytes).unwrap();
        let starts = set_slots(&nano(
            parsed.blocks(),
            parsed.header().data_region_start,
            parsed.header().data_region_size,
            16,
        ));
        assert_eq!(starts.len(), parsed.blocks().len());
        assert!(parsed.blocks()[0].continuation);
        assert!(parsed.blocks()[1].continuation);
        assert!(!parsed.blocks()[2].continuation);
    }

    #[test]
    fn checkpoint_select_covers_first_last_and_every_ordinal() {
        let corpus = corpora()
            .into_iter()
            .find(|corpus| corpus.name == "many-tiny")
            .unwrap();
        let bytes = make_file(&corpus, 5);
        let parsed = parse_v06(&bytes).unwrap();
        let slots = set_slots(&nano(
            parsed.blocks(),
            parsed.header().data_region_start,
            parsed.header().data_region_size,
            32,
        ));
        for interval in CHECKPOINTS {
            let slot_count = parsed.header().data_region_size.div_ceil(32);
            let cp = checkpoints(
                &nano(
                    parsed.blocks(),
                    parsed.header().data_region_start,
                    parsed.header().data_region_size,
                    32,
                ),
                slot_count,
                interval,
            );
            for ordinal in 0..slots.len() as u64 {
                let bits = nano(
                    parsed.blocks(),
                    parsed.header().data_region_start,
                    parsed.header().data_region_size,
                    32,
                );
                let (start, next) =
                    select_checkpointed(&bits, &cp, slot_count, interval, ordinal).unwrap();
                assert_eq!(start, slots[ordinal as usize]);
                let expected_next = slots
                    .get(ordinal as usize + 1)
                    .copied()
                    .unwrap_or(slot_count);
                assert_eq!(next, expected_next);
            }
        }
    }

    #[test]
    fn directory_lookup_returns_the_same_validated_range_as_canonical_scan() {
        let corpus = corpora()
            .into_iter()
            .find(|corpus| corpus.name == "many-tiny")
            .unwrap();
        let bytes = make_file(&corpus, 4);
        let parsed = parse_v06(&bytes).unwrap();
        let directory = make_directory(parsed.blocks());
        for key in 0..256u16 {
            assert_eq!(
                range_checksum(lookup_range(parsed.blocks(), key)),
                directory_checksum(&directory, key)
            );
        }
    }

    #[test]
    fn checked_nano_arithmetic_rejects_invalid_and_huge_regions() {
        assert!(checked_nano_len(u64::MAX, 8).is_err());
        assert!(checked_nano_len(1, 0).is_err());
        assert!(checked_nano_len(1, 3).is_err());
        assert_eq!(checked_nano_len(0, 8).unwrap(), 0);
        assert_eq!(checked_nano_len(65, 8).unwrap(), 2);
    }
}

fn main() {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark-results/vbuf-navigation/step5a.csv".into());
    let mut samples = Vec::new();
    let mut geometry = BTreeMap::new();
    for base_shift in [3u8, 4, 5, 6, 7, 8] {
        let base_step = 1u64 << base_shift;
        for corpus in corpora() {
            let bytes = make_file(&corpus, base_shift);
            let parsed = parse_v06(&bytes).unwrap();
            let blocks = parsed.blocks();
            let payload_bytes = blocks.iter().map(|b| b.payload_len).sum();
            let bits = nano(
                blocks,
                parsed.header().data_region_start,
                parsed.header().data_region_size,
                base_step,
            );
            let slots = set_slots(&bits);
            let slot_count = parsed.header().data_region_size.div_ceil(base_step);
            let directory = make_directory(blocks);
            let directory_bytes = directory.len() * 32;
            let nano_bytes = bits.len();
            let serialized_nano = serialize_nano(&bits);
            let header_bytes = 24
                + blocks
                    .iter()
                    .map(|block| if block.count > 65535 { 16 } else { 8 })
                    .sum::<usize>();
            let padding_bytes = bytes.len() - payload_bytes as usize - header_bytes;
            geometry.insert((corpus.name, base_step), (header_bytes, padding_bytes));
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "canonical-validation",
                "parse-and-describe",
                measure(|| parse_v06(&bytes).unwrap().blocks().len() as u64),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "nano-construction",
                "reconstruct-from-validated-canonical",
                measure(|| {
                    nano(
                        blocks,
                        parsed.header().data_region_start,
                        parsed.header().data_region_size,
                        base_step,
                    )
                    .len() as u64
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "nano-deployment",
                "embedded-load",
                measure(|| nano_checksum(&deserialize_nano(&serialized_nano))),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "nano-deployment",
                "local-reconstruction",
                measure(|| {
                    nano_checksum(&nano(
                        blocks,
                        parsed.header().data_region_start,
                        parsed.header().data_region_size,
                        base_step,
                    ))
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "nano-deployment",
                "reconstructed-cache-load",
                measure(|| nano_checksum(&deserialize_nano(&serialized_nano))),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "canonical-block-traversal",
                "canonical",
                measure(|| blocks.iter().map(|b| payload_sum(&bytes, b)).sum()),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "physical-start-enumeration",
                "nano",
                measure(|| {
                    nano_start_sum(
                        &bits,
                        slot_count,
                        parsed.header().data_region_start,
                        base_step,
                    )
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                slots.len() as u64,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "nth-start-plus-next-boundary",
                "raw-nano-linear-select",
                measure(|| {
                    (0..slots.len() as u64)
                        .map(|ordinal| {
                            let start = raw_select(&bits, slot_count, ordinal).unwrap();
                            let next = (start + 1..slot_count)
                                .find(|candidate| bit_is_set(&bits, *candidate))
                                .unwrap_or(slot_count);
                            start ^ next
                        })
                        .sum()
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            for interval in CHECKPOINTS {
                let cp = checkpoints(&bits, slot_count, interval);
                let cp_bytes = cp.len() * 8;
                append_samples(
                    &mut samples,
                    corpus.name,
                    base_step,
                    "nth-start-plus-next-boundary",
                    match interval {
                        512 => "nano+checkpoint-512",
                        4096 => "nano+checkpoint-4096",
                        _ => "nano+checkpoint-65536",
                    },
                    measure(|| {
                        (0..slots.len() as u64)
                            .filter_map(|i| {
                                select_checkpointed(&bits, &cp, slot_count, interval, i)
                            })
                            .map(|(start, next)| start ^ next)
                            .sum()
                    }),
                    bytes.len(),
                    payload_bytes,
                    blocks.len(),
                    nano_bytes,
                    cp_bytes,
                    directory_bytes,
                    0,
                );
            }
            let sorted = directory;
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "directory-construction",
                "sorted-validated-ranges",
                measure(|| {
                    let mut candidate: Vec<(u16, u64, u64, u64)> = blocks
                        .iter()
                        .map(|b| (b.key_id, b.block_start, b.payload_start, b.payload_len))
                        .collect();
                    candidate.sort_unstable();
                    candidate.len() as u64
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "key-lookup",
                "canonical-linear-scan",
                measure(|| {
                    (0..256u16)
                        .map(|key| range_checksum(lookup_range(blocks, key)))
                        .sum()
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "key-lookup",
                "directory-binary-search",
                measure(|| {
                    (0..256u16)
                        .map(|key| directory_checksum(&sorted, key))
                        .sum()
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "parallel-payload-sum",
                "canonical-rayon",
                measure(|| blocks.par_iter().map(|b| payload_sum(&bytes, b)).sum()),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "parallel-payload-sum",
                "nano-guided-rayon",
                measure(|| {
                    (0..slot_count)
                        .into_par_iter()
                        .filter_map(|slot| {
                            if !bit_is_set(&bits, slot) {
                                return None;
                            }
                            let start = parsed.header().data_region_start + slot * base_step;
                            blocks
                                .iter()
                                .find(|block| block.block_start == start)
                                .map(|block| payload_sum(&bytes, block))
                        })
                        .sum()
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "parallel-payload-sum",
                "dynamic-queue-control",
                measure(|| {
                    blocks
                        .par_iter()
                        .map(|block| payload_sum(&bytes, block))
                        .sum()
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                0,
            );
            append_samples(
                &mut samples,
                corpus.name,
                base_step,
                "parallel-start-enumeration",
                "nano-rayon",
                measure(|| {
                    (0..slot_count)
                        .into_par_iter()
                        .filter(|slot| bit_is_set(&bits, *slot))
                        .map(|slot| parsed.header().data_region_start + slot * base_step)
                        .sum()
                }),
                bytes.len(),
                payload_bytes,
                blocks.len(),
                nano_bytes,
                0,
                directory_bytes,
                slots.len() as u64,
            );
        }
    }
    let path = std::path::Path::new(&output);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = File::create(path).unwrap();
    use std::io::Write;
    writeln!(file, "layout,base_step,operation,variant,sample,nanos,file_bytes,header_bytes,padding_bytes,payload_bytes,block_count,nano_bytes,checkpoint_bytes,directory_bytes,checksum").unwrap();
    for s in samples {
        let (header_bytes, padding_bytes) = geometry[&(s.layout, s.base_step)];
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            s.layout,
            s.base_step,
            s.operation,
            s.variant,
            s.sample,
            s.nanos,
            s.file_bytes,
            header_bytes,
            padding_bytes,
            s.payload_bytes,
            s.block_count,
            s.nano_bytes,
            s.checkpoint_bytes,
            s.directory_bytes,
            s.checksum
        )
        .unwrap();
    }
}
