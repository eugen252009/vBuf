use std::io::Write;
use xxhash_rust::xxh3::xxh3_64;

// --- STRATEGY PATTERN FÜR CHECKSUMS ---

pub trait ChecksumStrategy {
    fn compute(&self, data: &[u8]) -> u32;
    fn name(&self) -> &'static str;
}

pub struct Xxh3Strategy;
impl ChecksumStrategy for Xxh3Strategy {
    fn compute(&self, data: &[u8]) -> u32 {
        xxh3_64(data) as u32
    }
    fn name(&self) -> &'static str {
        "XXH3"
    }
}

pub struct NoChecksumStrategy;
impl ChecksumStrategy for NoChecksumStrategy {
    fn compute(&self, _data: &[u8]) -> u32 {
        0
    }
    fn name(&self) -> &'static str {
        "None"
    }
}

// --- TYPEN & ENUMS ---

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
    BLOB = 3,
    FLOT = 5,
    TRU = 6,
    FAL = 7,
}

#[derive(Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub start: usize,
    pub len: usize,
}

// --- VBUF CORE ---

pub struct VBuf {
    pub cells: Vec<Vec<u8>>,
    strategy: Box<dyn ChecksumStrategy>,
}

impl VBuf {
    pub fn empty(strategy: Box<dyn ChecksumStrategy>) -> Self {
        Self {
            cells: Vec::new(),
            strategy,
        }
    }

    pub fn stream_json<W: Write>(&self, input: &[u8], stream: &mut W) -> std::io::Result<()> {
        let mut lexer = VBufLexer::new(input);
        let first_token = lexer.next_token();
        lexer.parse_and_stream(stream, self, String::new(), first_token)
    }

    fn _create_cell(&self, s_type: u8, p_type: u8, key: &str, payload: &[u8]) -> Vec<u8> {
        let key_bytes = key.as_bytes();
        let safe_val_len = payload.len().min(65535) as u16;

        // Wir berechnen den Hash weiterhin vom Namen...
        let path_hash = xxh3_64(key_bytes) as u32;
        let combined_type = (s_type << 4) | (p_type & 0x0F);

        // ...aber im Header setzen wir key_len auf 0, da wir den String nicht speichern
        let header: u64 = (path_hash as u64)
        | ((combined_type as u64) << 32)
        | (0u64 << 40) // key_len ist jetzt 0
        | ((safe_val_len as u64) << 48);

        // Größe: 8 (Header) + 0 (Key) + Payload + 4 (CRC)
        let data_size = 8 + safe_val_len as usize + 4;
        let total_size = (data_size + 15) & !15;

        let mut cell = vec![0; total_size];
        cell[0..8].copy_from_slice(&header.to_le_bytes());

        // Payload kommt jetzt DIREKT nach dem Header (Offset 8)
        let val_end = 8 + safe_val_len as usize;
        cell[8..val_end].copy_from_slice(&payload[..safe_val_len as usize]);

        let crc = self.strategy.compute(&cell[..total_size - 4]);
        let crc_start = total_size - 4;
        cell[crc_start..total_size].copy_from_slice(&crc.to_le_bytes());

        cell
    }
}

// --- STREAMING LEXER LOGIK ---

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
                        self.parse_and_stream(stream, vbuf, new_prefix, val_token)?;
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

                    let new_prefix = format!("{}[{}]", prefix, index);
                    self.parse_and_stream(stream, vbuf, new_prefix, val_token)?;
                    index += 1;

                    if self.peek_token().kind == TokenKind::Comma {
                        self.next_token();
                    }
                }
            }
            TokenKind::String => {
                let val = &self.input
                    [current_token.start + 1..current_token.start + current_token.len - 1];
                let cell = vbuf._create_cell(Sem::STR as u8, Phys::BLOB as u8, &prefix, val);
                stream.write_all(&cell)?;
            }
            TokenKind::Number => {
                let val_bytes =
                    &self.input[current_token.start..current_token.start + current_token.len];
                let cell = vbuf._create_cell(Sem::NUM as u8, Phys::FLOT as u8, &prefix, val_bytes);
                stream.write_all(&cell)?;
            }
            TokenKind::True | TokenKind::False => {
                let b = if current_token.kind == TokenKind::True {
                    [1]
                } else {
                    [0]
                };
                let p_type = if current_token.kind == TokenKind::True {
                    Phys::TRU
                } else {
                    Phys::FAL
                };
                let cell = vbuf._create_cell(Sem::BOL as u8, p_type as u8, &prefix, &b);
                stream.write_all(&cell)?;
            }
            TokenKind::Null => {
                let cell = vbuf._create_cell(Sem::NUL as u8, Phys::NUL as u8, &prefix, &[]);
                stream.write_all(&cell)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn peek_token(&mut self) -> Token {
        let current_i = self.i;
        let t = self.next_token();
        self.i = current_i;
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
