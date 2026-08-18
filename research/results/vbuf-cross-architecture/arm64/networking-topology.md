# ARM64 Networking Topology

The SSH alias `pixel` reaches Termux sshd through an existing ADB forward:

```text
ADB device: 192.168.188.21:38339 initially; 192.168.188.21:39907 after reconnect
adb forward: tcp:10022 -> tcp:8022
ssh pixel: localhost:10022
```

Direct LAN HTTP from Termux to workstation `192.168.188.40:18765` timed out.
The existing controlled Range server was therefore exposed through an ADB
reverse, without changing vBuf or SourceSet behavior:

```sh
adb reverse tcp:18765 tcp:18765
```

The Pixel then used:

```text
http://127.0.0.1:18765/Qwen3-32B-Q8_0.vbuf
```

The server implementation is the preserved generic Range server at
`research/results/vbuf-cross-architecture/arm32/scaffolding/vbuf-arm32-range-server.py`.
ADB is transport plumbing only and is not part of the vBuf architecture.
