# Inference Cost Tables

| Model | Device | Seq lens | Scaling exp | Timing |
| --- | --- | --- | --- | --- |
| hybrid | mps | 64,128,256,512 | 1.078 | 64:16.80ms/30477.0tps, 128:43.30ms/23646.5tps, 256:81.54ms/25117.4tps, 512:164.25ms/24937.7tps |
| transformer | mps | 64,128,256,512 | 1.008 | 64:13.02ms/39323.2tps, 128:24.10ms/42488.2tps, 256:47.88ms/42772.1tps, 512:106.39ms/38499.1tps |
