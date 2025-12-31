## Install ZIG

```bash
curl https://raw.githubusercontent.com/tristanisham/zvm/master/install.sh | bash

zvm i 0.15.2
zvm use 0.15.2
```

## Dowload a safetensor file

### Direct safetensors url


```bash
wget https://huggingface.co/openai/gpt-oss-20b/resolve/main/original/model.safetensors
```

Require

```zig
const tensor_name = "block.0.attn.qkv.weight";
```

----


```bash
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
```

Require

```zig
const tensor_name = "model.layers.0.self_attn.q_proj.weight";
```


## Build dequantizer

gpt-oss-20b ready

```bash
zig build -Doptimize=ReleaseSafe
./zig-out/bin/dequantizer ./model.safetensors
```

## ZLS

```bash
git clone --depth 1 --branch 0.15.1 https://github.com/zigtools/zls
cd zls
zig build -Doptimize=ReleaseSafe
```
