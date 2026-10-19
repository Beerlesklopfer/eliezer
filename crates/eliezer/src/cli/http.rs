//! Minimal blocking HTTP/1.1 client for the local service.
//!
//! Decision: written over `std::net::TcpStream` instead of `reqwest`. The service speaks plain
//! HTTP on a local address; the client needs only GET/POST with JSON, `Content-Length` and
//! chunked responses and line-wise reading of SSE. This avoids a TLS stack and an async runtime
//! in the command line path (reqwest blocking would pull in about a hundred more crates).
//! `https://` is therefore refused with a clear message.
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    time::Duration,
};

#[derive(Debug)]
pub enum HttpError {
    /// The connection could not be established (refused, unreachable, timeout, unknown host).
    Unreachable(String),
    /// The connection broke or the answer was malformed.
    Protocol(String),
    /// Something answered, but it is not Eliezer (no `Server: eliezer/...` header); LM Studio
    /// listens on the same default port. Carries what the server called itself.
    NotEliezer(String),
}

/// Address of the service, parsed from `http://host:port`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
}
impl Endpoint {
    pub fn parse(url: &str) -> Result<Self, String> {
        let rest = if let Some(rest) = url.strip_prefix("http://") {
            rest
        } else if url.starts_with("https://") {
            return Err("Only plain http:// is supported (the service speaks HTTP)".to_owned());
        } else if url.contains("://") {
            return Err("The address must look like http://HOST:PORT".to_owned());
        } else {
            url
        };
        let rest = rest.trim_end_matches('/');
        if rest.is_empty() || rest.contains('/') || rest.contains('@') {
            return Err("The address must look like http://HOST:PORT".to_owned());
        }
        let (host, port) = if let Some(v6) = rest.strip_prefix('[') {
            let (host, tail) = v6.split_once(']').ok_or("Invalid IPv6 address")?;
            let port = match tail.strip_prefix(':') {
                Some(p) => p,
                None if tail.is_empty() => "80",
                None => return Err("Invalid address".to_owned()),
            };
            (host, port)
        } else {
            rest.rsplit_once(':').unwrap_or((rest, "80"))
        };
        let port: u16 = port.parse().map_err(|_| "Invalid port in the address")?;
        if host.is_empty() || host.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err("Invalid host in the address".to_owned());
        }
        Ok(Self {
            host: host.to_owned(),
            port,
        })
    }
    /// Display form with scheme, IPv6 in brackets.
    pub fn url(&self) -> String {
        if self.host.contains(':') {
            format!("http://[{}]:{}", self.host, self.port)
        } else {
            format!("http://{}:{}", self.host, self.port)
        }
    }
}

pub struct Client {
    pub endpoint: Endpoint,
    /// Bearer key; only ever written into the Authorization header.
    key: Option<String>,
}

enum Mode {
    Length(u64),
    Chunked {
        left: u64,
        after_data: bool,
        done: bool,
    },
    Eof,
}
/// Response body: decodes `Content-Length`, chunked or read-until-close framing.
pub struct Body {
    inner: BufReader<TcpStream>,
    mode: Mode,
}
impl Read for Body {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let Self { inner, mode } = self;
        match mode {
            Mode::Eof => inner.read(buf),
            Mode::Length(left) => {
                if *left == 0 {
                    return Ok(0);
                }
                let max = usize::try_from(*left).unwrap_or(usize::MAX).min(buf.len());
                let n = inner.read(&mut buf[..max])?;
                if n == 0 {
                    return Err(io::ErrorKind::UnexpectedEof.into());
                }
                *left -= n as u64;
                Ok(n)
            }
            Mode::Chunked {
                left,
                after_data,
                done,
            } => {
                if *done {
                    return Ok(0);
                }
                if *left == 0 {
                    let mut line = String::new();
                    if *after_data {
                        inner.read_line(&mut line)?; // CRLF after the chunk data
                        line.clear();
                    }
                    if inner.read_line(&mut line)? == 0 {
                        return Err(io::ErrorKind::UnexpectedEof.into());
                    }
                    let size = line.trim().split(';').next().unwrap_or("");
                    *left = u64::from_str_radix(size, 16).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "bad chunk size")
                    })?;
                    if *left == 0 {
                        // trailers up to the empty line
                        loop {
                            line.clear();
                            if inner.read_line(&mut line)? == 0 || line.trim().is_empty() {
                                break;
                            }
                        }
                        *done = true;
                        return Ok(0);
                    }
                }
                let max = usize::try_from(*left).unwrap_or(usize::MAX).min(buf.len());
                let n = inner.read(&mut buf[..max])?;
                if n == 0 {
                    return Err(io::ErrorKind::UnexpectedEof.into());
                }
                *left -= n as u64;
                *after_data = true;
                Ok(n)
            }
        }
    }
}

pub struct Response {
    pub status: u16,
    body: Body,
}
impl Response {
    /// The whole body as text (at most 64 MiB).
    pub fn text(self) -> io::Result<String> {
        let mut bytes = Vec::new();
        self.body.take(64 << 20).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    /// The body line by line (for SSE); lines are returned without the line ending.
    pub fn lines(self) -> impl Iterator<Item = io::Result<String>> {
        let mut reader = BufReader::new(self.body);
        std::iter::from_fn(move || {
            let mut raw = Vec::new();
            match reader.read_until(b'\n', &mut raw) {
                Ok(0) => None,
                Ok(_) => {
                    while matches!(raw.last(), Some(b'\n' | b'\r')) {
                        raw.pop();
                    }
                    Some(Ok(String::from_utf8_lossy(&raw).into_owned()))
                }
                Err(e) => Some(Err(e)),
            }
        })
    }
}

impl Client {
    pub fn new(endpoint: Endpoint, key: Option<String>) -> Self {
        Self { endpoint, key }
    }
    pub fn has_key(&self) -> bool {
        self.key.is_some()
    }

    /// Sends one request. `read_timeout` limits the wait for each piece of the answer.
    pub fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
        read_timeout: Duration,
    ) -> Result<Response, HttpError> {
        let addrs = (self.endpoint.host.as_str(), self.endpoint.port)
            .to_socket_addrs()
            .map_err(|e| HttpError::Unreachable(e.to_string()))?;
        let mut last = "no address".to_owned();
        let mut stream = None;
        for addr in addrs {
            match TcpStream::connect_timeout(&addr, Duration::from_secs(3)) {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) => last = e.to_string(),
            }
        }
        let mut stream = stream.ok_or(HttpError::Unreachable(last))?;
        let proto = |e: io::Error| HttpError::Protocol(e.to_string());
        stream.set_read_timeout(Some(read_timeout)).map_err(proto)?;
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .map_err(proto)?;
        let host = if self.endpoint.host.contains(':') {
            format!("[{}]:{}", self.endpoint.host, self.endpoint.port)
        } else {
            format!("{}:{}", self.endpoint.host, self.endpoint.port)
        };
        let mut head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: eliezer/{}\r\nAccept: application/json, text/event-stream\r\nConnection: close\r\n",
            env!("CARGO_PKG_VERSION")
        );
        if let Some(key) = &self.key {
            head += &format!("Authorization: Bearer {key}\r\n");
        }
        if let Some(body) = body {
            head += &format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                body.len()
            );
        }
        head += "\r\n";
        stream.write_all(head.as_bytes()).map_err(proto)?;
        if let Some(body) = body {
            stream.write_all(body.as_bytes()).map_err(proto)?;
        }
        stream.flush().map_err(proto)?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(proto)?;
        let status: u16 = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| HttpError::Protocol("The answer is not valid HTTP".to_owned()))?;
        let (mut length, mut chunked) = (None, false);
        let mut server: Option<String> = None;
        for _ in 0..200 {
            line.clear();
            if reader.read_line(&mut line).map_err(proto)? == 0 {
                return Err(HttpError::Protocol("The answer ended early".to_owned()));
            }
            let header = line.trim_end();
            if header.is_empty() {
                let mode = if chunked {
                    Mode::Chunked {
                        left: 0,
                        after_data: false,
                        done: false,
                    }
                } else if let Some(n) = length {
                    Mode::Length(n)
                } else if status == 204 || status == 304 || (100..200).contains(&status) {
                    Mode::Length(0)
                } else {
                    Mode::Eof
                };
                if !server
                    .as_deref()
                    .is_some_and(|s| s.to_ascii_lowercase().starts_with("eliezer"))
                {
                    return Err(HttpError::NotEliezer(
                        server.unwrap_or_else(|| "no Server header".to_owned()),
                    ));
                }
                return Ok(Response {
                    status,
                    body: Body {
                        inner: reader,
                        mode,
                    },
                });
            }
            if let Some((name, value)) = header.split_once(':') {
                let (name, value) = (name.trim().to_ascii_lowercase(), value.trim());
                if name == "server" {
                    server = Some(value.to_owned());
                } else if name == "content-length" {
                    length = value.parse().ok();
                } else if name == "transfer-encoding"
                    && value.to_ascii_lowercase().contains("chunked")
                {
                    chunked = true;
                }
            }
        }
        Err(HttpError::Protocol("Too many header lines".to_owned()))
    }
}
