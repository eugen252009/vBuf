# RISC-V Networking Topology

The Orange Pi RV2 reached the workstation directly over LAN:

```text
Orange Pi end0: 192.168.188.42
Workstation:    192.168.188.40
HTTP server:    192.168.188.40:18765
```

The source URL used by RISC-V was:

```text
http://192.168.188.40:18765/Qwen3-32B-Q8_0.vbuf
```

No ADB, reverse tunnel, or Android-specific transport was used.
