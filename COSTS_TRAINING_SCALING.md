# Training Scaling Tables

| Model | Device | Seq lens | Timing |
| --- | --- | --- | --- |
| hybrid | mps | 128,256,512 | 128:196.76ms/2602.1tps/585826560B, 256:179.12ms/5716.8tps/586019072B, 512:317.72ms/6445.9tps/586608896B |
| transformer | mps | 128,256,512 | 128:85.26ms/6005.4tps/777194752B, 256:117.64ms/8704.7tps/777779968B, 512:226.00ms/9061.9tps/777779968B |
