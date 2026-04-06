use std::io::Write;
use xxhash_rust::xxh3::xxh3_64;

// --- ARENA ---

pub struct DynamicArena {
    buffer: Vec<u8>,
}

impl DynamicArena {
    pub fn new(initial_size: usize) -> Self {
        Self {
            buffer: vec![0u8; initial_size],
        }
    }

    pub fn get_mut(&mut self, required_size: usize) -> &mut [u8] {
        if required_size > self.buffer.len() {
            let new_size = (required_size + 4095) & !4095;
            self.buffer.resize(new_size, 0);
            // eprintln!("[vBuf] Arena auf {} Bytes vergrößert", new_size);
        }
        &mut self.buffer[..required_size]
    }
}

// --- STRATEGY PATTERN ---

pub trait ChecksumStrategy {
    fn compute(&self, data: &[u8]) -> u32;
}

pub struct Xxh3Strategy;
impl ChecksumStrategy for Xxh3Strategy {
    fn compute(&self, data: &[u8]) -> u32 {
        xxh3_64(data) as u32
    }
}

// --- TYPEN ---

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TokenKind {
    LCurly,
    RCurly,
    LBracket,
    RBracket,
    Colon,
    Comma,
    String,
    Number,
    True,
    False,
    Null,
    EndOfFile,
}

pub struct Token {
    pub kind: TokenKind,
    pub start: usize,
    pub len: usize,
}

#[repr(u8)]
pub enum Sem {
    NUM = 0,
    STR = 1,
    BOL = 2,
    NUL = 3,
    ARR = 4,
    OBJ = 5,
}
#[repr(u8)]
pub enum Phys {
    NUL = 0,
    SMI = 1,
    BLOB = 3,
    FLOT = 5,
    TRU = 6,
    FAL = 7,
}

// --- VBUF CORE ---

pub struct VBuf {
    pub strategy: Box<dyn ChecksumStrategy>,
}

impl VBuf {
    pub fn new(strategy: Box<dyn ChecksumStrategy>) -> Self {
        Self { strategy }
    }

    pub fn stream_json<W: Write>(
        &self,
        input: &[u8],
        stream: &mut W,
        arena: &mut DynamicArena,
    ) -> std::io::Result<()> {
        stream.write_all(b"VBUF0.3\0")?;
        stream.write_all(&[0u8; 8])?;

        let mut lexer = VBufLexer::new(input);
        let first_token = lexer.next_token();

        lexer.parse_and_stream(stream, self, String::new(), first_token, arena)
    }

    pub fn write_cell_to_arena<'a>(
        &self,
        arena: &'a mut DynamicArena,
        s_type: u8,
        p_type: u8,
        key: &str,
        payload: &[u8],
        inline_val: Option<u32>,
    ) -> &'a [u8] {
        let path_hash = xxh3_64(key.as_bytes()) as u32;
        let combined_type = (s_type << 4) | (p_type & 0x0F);
        let val_len = if inline_val.is_some() {
            4u16
        } else {
            payload.len() as u16
        };

        let total_size = (8 + val_len as usize + 4 + 15) & !15;
        let buffer = arena.get_mut(total_size);

        let header: u64 =
            (path_hash as u64) | ((combined_type as u64) << 32) | ((val_len as u64) << 48);

        buffer[0..8].copy_from_slice(&header.to_le_bytes());

        if let Some(val) = inline_val {
            buffer[8..12].copy_from_slice(&val.to_le_bytes());
        } else {
            buffer[8..8 + payload.len()].copy_from_slice(payload);
        }

        let crc_pos = total_size - 4;
        for i in (8 + val_len as usize)..crc_pos {
            buffer[i] = 0;
        }
        let crc = self.strategy.compute(&buffer[..crc_pos]);
        buffer[crc_pos..total_size].copy_from_slice(&crc.to_le_bytes());

        &buffer[..total_size]
    }
}

// --- LEXER ---

pub struct VBufLexer<'a> {
    input: &'a [u8],
    i: usize,
}

impl<'a> VBufLexer<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Self { input, i: 0 }
    }

    pub fn parse_and_stream<W: Write>(
        &mut self,
        stream: &mut W,
        vbuf: &VBuf,
        prefix: String,
        current_token: Token,
        arena: &mut DynamicArena,
    ) -> std::io::Result<()> {
        match current_token.kind {
            TokenKind::LCurly => loop {
                let key_token = self.next_token();
                if key_token.kind == TokenKind::RCurly || key_token.kind == TokenKind::EndOfFile {
                    break;
                }
                if key_token.kind == TokenKind::String {
                    let key_str = std::str::from_utf8(
                        &self.input[key_token.start + 1..key_token.start + key_token.len - 1],
                    )
                    .unwrap_or("");
                    let new_prefix = if prefix.is_empty() {
                        key_str.to_string()
                    } else {
                        format!("{}.{}", prefix, key_str)
                    };
                    if self.next_token().kind == TokenKind::Colon {
                        let val_token = self.next_token();
                        self.parse_and_stream(stream, vbuf, new_prefix, val_token, arena)?;
                    }
                }
                if self.peek_token().kind == TokenKind::Comma {
                    self.next_token();
                }
            },
            TokenKind::LBracket => {
                let mut index = 0;
                loop {
                    let val_token = self.next_token();
                    if val_token.kind == TokenKind::RBracket
                        || val_token.kind == TokenKind::EndOfFile
                    {
                        break;
                    }
                    self.parse_and_stream(
                        stream,
                        vbuf,
                        format!("{}[{}]", prefix, index),
                        val_token,
                        arena,
                    )?;
                    index += 1;
                    if self.peek_token().kind == TokenKind::Comma {
                        self.next_token();
                    }
                }
            }
            TokenKind::String => {
                let val = &self.input
                    [current_token.start + 1..current_token.start + current_token.len - 1];
                stream.write_all(vbuf.write_cell_to_arena(
                    arena,
                    Sem::STR as u8,
                    Phys::BLOB as u8,
                    &prefix,
                    val,
                    None,
                ))?;
            }
            TokenKind::Number => {
                let val_bytes =
                    &self.input[current_token.start..current_token.start + current_token.len];
                let val_str = std::str::from_utf8(val_bytes).unwrap_or("0");
                let cell = if let Ok(num) = val_str.parse::<u16>() {
                    vbuf.write_cell_to_arena(
                        arena,
                        Sem::NUM as u8,
                        Phys::SMI as u8,
                        &prefix,
                        &[],
                        Some(num as u32),
                    )
                } else if let Ok(f_num) = val_str.parse::<f64>() {
                    vbuf.write_cell_to_arena(
                        arena,
                        Sem::NUM as u8,
                        Phys::FLOT as u8,
                        &prefix,
                        &f_num.to_le_bytes(),
                        None,
                    )
                } else {
                    vbuf.write_cell_to_arena(
                        arena,
                        Sem::NUM as u8,
                        Phys::BLOB as u8,
                        &prefix,
                        val_bytes,
                        None,
                    )
                };
                stream.write_all(cell)?;
            }
            TokenKind::True | TokenKind::False => {
                let val = if current_token.kind == TokenKind::True {
                    1
                } else {
                    0
                };
                let phys = if val == 1 { Phys::TRU } else { Phys::FAL };
                stream.write_all(vbuf.write_cell_to_arena(
                    arena,
                    Sem::BOL as u8,
                    phys as u8,
                    &prefix,
                    &[],
                    Some(val),
                ))?;
            }
            TokenKind::Null => {
                stream.write_all(vbuf.write_cell_to_arena(
                    arena,
                    Sem::NUL as u8,
                    Phys::NUL as u8,
                    &prefix,
                    &[],
                    None,
                ))?;
            }
            _ => {}
        }
        Ok(())
    }

    fn peek_token(&mut self) -> Token {
        let i = self.i;
        let t = self.next_token();
        self.i = i;
        t
    }

    fn next_token(&mut self) -> Token {
        while self.i < self.input.len() && self.input[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
        if self.i >= self.input.len() {
            return Token {
                kind: TokenKind::EndOfFile,
                start: self.i,
                len: 0,
            };
        }
        let start = self.i;
        let kind = match self.input[self.i] {
            b'{' => {
                self.i += 1;
                TokenKind::LCurly
            }
            b'}' => {
                self.i += 1;
                TokenKind::RCurly
            }
            b'[' => {
                self.i += 1;
                TokenKind::LBracket
            }
            b']' => {
                self.i += 1;
                TokenKind::RBracket
            }
            b':' => {
                self.i += 1;
                TokenKind::Colon
            }
            b',' => {
                self.i += 1;
                TokenKind::Comma
            }
            b'"' => {
                self.i += 1;
                while self.i < self.input.len() {
                    if self.input[self.i] == b'\\' {
                        self.i += 2;
                    } else if self.input[self.i] == b'"' {
                        self.i += 1;
                        break;
                    } else {
                        self.i += 1;
                    }
                }
                TokenKind::String
            }
            b'0'..=b'9' | b'-' => {
                while self.i < self.input.len()
                    && (self.input[self.i].is_ascii_digit()
                        || matches!(self.input[self.i], b'.' | b'e' | b'E' | b'-' | b'+'))
                {
                    self.i += 1;
                }
                TokenKind::Number
            }
            b't' if self.input.get(self.i..self.i + 4) == Some(b"true") => {
                self.i += 4;
                TokenKind::True
            }
            b'f' if self.input.get(self.i..self.i + 5) == Some(b"false") => {
                self.i += 5;
                TokenKind::False
            }
            b'n' if self.input.get(self.i..self.i + 4) == Some(b"null") => {
                self.i += 4;
                TokenKind::Null
            }
            _ => {
                self.i += 1;
                TokenKind::Null
            }
        };
        Token {
            kind,
            start,
            len: self.i - start,
        }
    }
}
