# MusicVAE

## 데이터 전처리 + 학습

```bash
python main.py
```



## 전처리

__Groove Midi Dataset__

| Pitch |  Roland Mapping  |   GM Mapping   |    Paper Mapping    | Frequency |
| :---: | :--------------: | :------------: | :-----------------: | --------: |
|  36   |       Kick       |  Bass Drum 1   |      Bass (36)      |     88067 |
|  38   |   Snare (Head)   | Acoustic Snare |     Snare (38)      |    102787 |
|  40   |   Snare (Rim)    | Electric Snare |     Snare (38)      |     22262 |
|  37   |  Snare X-Stick   |   Side Stick   |     Snare (38)      |      9696 |
|  48   |      Tom 1       |   Hi-Mid Tom   |    High Tom (50)    |     13145 |
|  50   |   Tom 1 (Rim)    |    High Tom    |    High Tom (50)    |      1561 |
|  45   |      Tom 2       |    Low Tom     |  Low-Mid Tom (47)   |      3935 |
|  47   |   Tom 2 (Rim)    |  Low-Mid Tom   |  Low-Mid Tom (47)   |      1322 |
|  43   |   Tom 3 (Head)   | High Floor Tom | High Floor Tom (43) |     11260 |
|  58   |   Tom 3 (Rim)    |   Vibraslap    | High Floor Tom (43) |      1003 |
|  46   |  HH Open (Bow)   |  Open Hi-Hat   |  Open Hi-Hat (46)   |      3905 |
|  26   |  HH Open (Edge)  |      N/A       |  Open Hi-Hat (46)   |     10243 |
|  42   | HH Closed (Bow)  | Closed Hi-Hat  | Closed Hi-Hat (42)  |     31691 |
|  22   | HH Closed (Edge) |      N/A       | Closed Hi-Hat (42)  |     34764 |
|  44   |     HH Pedal     |  Pedal Hi-Hat  | Closed Hi-Hat (42)  |     52343 |
|  49   |  Crash 1 (Bow)   | Crash Cymbal 1 |  Crash Cymbal (49)  |       720 |
|  55   |  Crash 1 (Edge)  | Splash Cymbal  |  Crash Cymbal (49)  |      5567 |
|  57   |  Crash 2 (Bow)   | Crash Cymbal 2 |  Crash Cymbal (49)  |      1832 |
|  52   |  Crash 2 (Edge)  | Chinese Cymbal |  Crash Cymbal (49)  |      1046 |
|  51   |    Ride (Bow)    | Ride Cymbal 1  |  Ride Cymbal (51)   |     43847 |
|  59   |   Ride (Edge)    | Ride Cymbal 2  |  Ride Cymbal (51)   |      2220 |
|  53   |   Ride (Bell)    |   Ride Bell    |  Ride Cymbal (51)   |      5567 |

- pitch 값에 따라 9개의 클래스로 mapping

- seq_len = 16 * 4마디 = 64, 9개 클래스의 조합으로 512dim => (64, 512) shape 의 Note sparse matrix 생성



## 모델

__Encoder__ : 양방향 LSTM

__Conductor__: 단방향 LSTM

__Decoder__: 단방향 LSTM



## 학습

- DataLoader를 통해 (B, 64, 512) shape의 데이터를 input으로 사용
- learning rate: 1e-4
- optimizer: Adam

- scheduler: CosineAnnealingLR(eta_min = 1e-6)



## 생성

```bash
python main.py --mode=test
```

- mu=0, std=1인 normal 분포에서 sample 개수 n에 따른 z 추출
- z => conductor => decoder 를 통해 4마디 sample output 생성 => (n, 64, 612)
- sample 희소행렬은 midi 파일로 변환 후 저장

